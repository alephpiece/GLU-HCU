#include "gpu-compat.h"
#include "numeric.h"
#include "update-plan.h"
#include <iostream>
#include "symbolic.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cstdint>

using namespace std;

template<UpdateMode Mode, bool Perturb = false>
__global__ void RL(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned* __restrict__ csr_r_ptr_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const int* __restrict__ level_idx_dev,
        REAL* __restrict__ tmpMem,
        const unsigned n,
        const int levelHead,
        const int inLevPos,
        const size_t value_stride = 0,
        const size_t scratch_stride = 0,
        const REAL pert = 0,
        const uint64_t* __restrict__ map_bases = nullptr,
        const unsigned* __restrict__ map_destinations = nullptr,
        const unsigned* __restrict__ map_pivots = nullptr)
{
    val_dev += size_t(blockIdx.y) * value_stride;
    if constexpr (Mode == UpdateMode::Scan)
        tmpMem += size_t(blockIdx.y) * scratch_stride;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / GLU_GROUP_SIZE;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    if constexpr (Perturb) {
        if (tid == 0 && fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
            val_dev[l_col_ptr_dev[currentCol]] = pert;
        __syncthreads();
    }
    // Preserve the original scan path, including its empty-column behavior.
    if constexpr (Mode == UpdateMode::Mapped)
        if (currentLColSize == 0) return;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            if constexpr (Mode == UpdateMode::Scan) {
                const unsigned ridx = sym_r_idx_dev[currentLPos + offset];
                tmpMem[size_t(bid)*n + ridx]= val_dev[currentLPos + offset];
            }
        }
        offset += blockDim.x;
    }
    // Same-level sources cannot update one another (symbolic U/L dependencies).
    // Publish normalized L before the block reads it directly in mapped mode.
    __syncthreads();

    if constexpr (Mode == UpdateMode::Mapped) {
        const unsigned first_target = csr_diag_ptr_dev[currentCol] + 1;
        for (unsigned q=first_target+wid; q<csr_r_ptr_dev[currentCol+1]; q+=blockDim.x/GLU_GROUP_SIZE) {
            const REAL pivot = val_dev[map_pivots[q]];
            const uint64_t base = map_bases[currentCol] + uint64_t(q-first_target)*currentLColSize;
            for (unsigned i=tid % GLU_GROUP_SIZE; i<currentLColSize; i+=GLU_GROUP_SIZE) {
                const unsigned source = l_col_ptr_dev[currentCol] + 1 + i;
                const REAL factor = val_dev[source];
                if (factor != REAL(0))
                    atomicAdd(&val_dev[map_destinations[base+i]], -factor*pivot);
            }
        }
    } else {
        //broadcast to submatrix
        const unsigned subColPos = csr_diag_ptr_dev[currentCol] + wid + 1;
        const unsigned subMatSize = csr_r_ptr_dev[currentCol + 1] - csr_diag_ptr_dev[currentCol] - 1;
        unsigned subCol;
        const int tidInWarp = threadIdx.x % GLU_GROUP_SIZE;
        unsigned subColElem = 0;

        int woffset = 0;
        while (subMatSize > woffset)
        {
            if (wid + woffset < subMatSize)
            {
                subCol = csr_c_idx_dev[subColPos + woffset];
                const unsigned begin = sym_c_ptr_dev[subCol];
                const unsigned count = sym_c_ptr_dev[subCol + 1] - begin;
                REAL pivot = 0;
                for (unsigned offset = 0; offset < count; offset += GLU_GROUP_SIZE)
                {
                    const bool active = tidInWarp + offset < count;
                    subColElem = begin + tidInWarp + offset;
                    const unsigned ridx = active ? sym_r_idx_dev[subColElem] : 0;
                    const bool is_pivot = active && ridx == currentCol;
                    const REAL value = is_pivot ? val_dev[subColElem] : REAL(0);
                    const unsigned mask = gpuBallot32(is_pivot);
                    if (mask)
                        pivot = gpuShuffle32(value, __ffs(mask) - 1);
                    // Sorted row indices put the pivot before all rows using it.
                    // Keep it in registers across chunks; no implicit warp sync.
                    if (active && ridx > currentCol) {
                        const REAL factor = tmpMem[ridx+n*size_t(bid)];
                        // Avoid zero-contribution writes to another column's pivot.
                        if (factor != REAL(0))
                            atomicAdd(&val_dev[subColElem], -factor*pivot);
                    }
                }
            }
            woffset += blockDim.x/GLU_GROUP_SIZE;
        }
    }

    if constexpr (Mode == UpdateMode::Scan) {
        __syncthreads();
        //Clear tmpMem
        offset = 0;
        while (currentLColSize > offset)
        {
            if (tid + offset < currentLColSize)
            {
                unsigned ridx = sym_r_idx_dev[currentLPos + offset];
                tmpMem[size_t(bid)*n + ridx]= 0;
            }
            offset += blockDim.x;
        }
    }
}

template<bool Perturb = false, bool UseScratch = true, bool Level = false>
__global__ void RL_onecol_factorizeCurrentCol(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned column,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const size_t value_stride = 0,
        const size_t scratch_stride = 0,
        const REAL pert = 0,
        const int* __restrict__ level_idx = nullptr,
        const int level_begin = 0)
{
    const unsigned currentCol = Level ? unsigned(level_idx[level_begin+blockIdx.x]) : column;
    val_dev += size_t(blockIdx.y) * value_stride;
    if constexpr (UseScratch)
        tmpMem += size_t(blockIdx.y) * scratch_stride;
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    if constexpr (Perturb) {
        if (tid == 0 && fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
            val_dev[l_col_ptr_dev[currentCol]] = pert;
        __syncthreads();
    }

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            if constexpr (UseScratch) {
                const unsigned ridx = sym_r_idx_dev[currentLPos + offset];
                tmpMem[size_t(stream) * n + ridx]= val_dev[currentLPos + offset];
            }
        }
        offset += blockDim.x;
    }
}

template<UpdateMode Mode>
__global__ void RL_onecol_updateSubmat(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const size_t value_stride = 0,
        const size_t scratch_stride = 0,
        const unsigned* __restrict__ l_col_ptr_dev = nullptr,
        const uint64_t* __restrict__ map_bases = nullptr,
        const unsigned* __restrict__ map_destinations = nullptr,
        const unsigned* __restrict__ map_pivots = nullptr)
{
    val_dev += size_t(blockIdx.y) * value_stride;
    if constexpr (Mode == UpdateMode::Scan)
        tmpMem += size_t(blockIdx.y) * scratch_stride;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ REAL s;

    if constexpr (Mode == UpdateMode::Mapped) {
        const unsigned lower = l_col_ptr_dev[currentCol]+1;
        const unsigned count = sym_c_ptr_dev[currentCol+1]-lower;
        if (!count) return;
        const unsigned q = csr_diag_ptr_dev[currentCol]+1+bid;
        const uint64_t base = map_bases[currentCol]+uint64_t(bid)*count;
        if (tid == 0) s = val_dev[map_pivots[q]];
        __syncthreads();
        for (unsigned i=tid; i<count; i+=blockDim.x) {
            const REAL factor = val_dev[lower+i];
            if (factor != REAL(0))
                atomicAdd(&val_dev[map_destinations[base+i]], -factor*s);
        }
    } else {
        //broadcast to submatrix
        const unsigned subColPos = csr_diag_ptr_dev[currentCol] + bid + 1;
        unsigned subCol;
        unsigned subColElem = 0;

        int offset = 0;
        subCol = csr_c_idx_dev[subColPos];
        if (tid == 0)
            s = 0;
        __syncthreads();
        while(offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol])
        {
            bool active = (tid + offset < sym_c_ptr_dev[subCol + 1] - sym_c_ptr_dev[subCol]);
            unsigned ridx = 0;
            if (active)
            {
                subColElem = sym_c_ptr_dev[subCol] + tid + offset;
                ridx = sym_r_idx_dev[subColElem];

                if (ridx == currentCol)
                {
                    s = val_dev[subColElem];
                }
            }
            // Every thread must reach both barriers; gating __syncthreads by "active"
            // can deadlock when sub-column nnz is smaller than blockDim.x.
            __syncthreads();
            if (active && ridx > currentCol)
            {
                const REAL factor = tmpMem[size_t(stream) * n + ridx];
                if (factor != REAL(0))
                    atomicAdd(&val_dev[subColElem], -factor * s);
            }
            __syncthreads();
            offset += blockDim.x;
        }
    }
}

// Deferred small levels expose source-target parallelism without a separate factor launch.
template<bool Perturb = false>
__global__ void mapped_level_update(
        REAL* __restrict__ values, size_t stride,
        const unsigned* __restrict__ col,
        const unsigned* __restrict__ diag,
        const unsigned* __restrict__ csr_diag,
        const int* __restrict__ level_idx,
        const uint64_t* __restrict__ prefix, int begin, int end,
        const uint64_t* __restrict__ bases,
        const unsigned* __restrict__ destinations,
        const unsigned* __restrict__ pivots, REAL threshold=0)
{
    values += size_t(blockIdx.y)*stride;
    __shared__ unsigned lower, count;
    __shared__ uint64_t base;
    __shared__ REAL pivot;
    __shared__ REAL divisor;
    if (threadIdx.x == 0) {
        const uint64_t rank=prefix[begin]+blockIdx.x;
        // upper_bound skips zero-target columns, including repeated prefixes.
        int first=begin, last=end;
        while (first<last) {
            const int mid=first+(last-first)/2;
            if (prefix[mid+1]<=rank) first=mid+1;
            else last=mid;
        }
        const unsigned source=unsigned(level_idx[first]);
        const uint64_t target=rank-prefix[first];
        lower=diag[source]+1;
        count=col[source+1]-lower;
        base=bases[source]+target*count;
        pivot=values[pivots[csr_diag[source]+1+target]];
        divisor=values[diag[source]];
        if constexpr(Perturb) if(fabs(divisor)<threshold) divisor=threshold;
    }
    __syncthreads();
    for (unsigned i=threadIdx.x; i<count; i+=blockDim.x) {
        REAL factor=values[lower+i];
        factor/=divisor;
        if (factor != REAL(0))
            atomicAdd(&values[destinations[base+i]], -factor*pivot);
    }
}

// Adjacent lanes access one symbolic position across independent value copies.
template<bool Perturb = false,bool Normalized = false>
__global__ void mapped_level_update_soa(
        REAL* values,unsigned copies,const unsigned* col,const unsigned* diag,
        const unsigned* csr_diag,const int* order,const uint64_t* prefix,int begin,int end,
        const uint64_t* bases,const unsigned* destinations,const unsigned* pivots,REAL threshold,unsigned value_stride,
        unsigned copy_bits)
{
    __shared__ unsigned lower,count,diagonal,pivot_position;
    __shared__ uint64_t base;
    if(threadIdx.x==0) {
        const uint64_t rank=prefix[begin]+blockIdx.x;
        int first=begin,last=end;
        while(first<last) {
            const int mid=first+(last-first)/2;
            if(prefix[mid+1]<=rank) first=mid+1; else last=mid;
        }
        const unsigned source=unsigned(order[first]);
        const uint64_t target=rank-prefix[first];
        diagonal=diag[source];lower=diagonal+1;count=col[source+1]-lower;
        base=bases[source]+target*count;
        pivot_position=pivots[csr_diag[source]+1+target];
    }
    __syncthreads();
    const unsigned copy_lanes=1u<<copy_bits,row_slots=blockDim.x>>copy_bits;
    const unsigned copy=blockIdx.y*copy_lanes+(threadIdx.x&(copy_lanes-1));
    if(copy>=copies) return;
    const size_t copy_base=copy;
    const unsigned step=value_stride;
    REAL divisor=REAL(1);
    if constexpr(!Normalized) {
        divisor=values[size_t(diagonal)*step+copy_base];
        if constexpr(Perturb) if(fabs(divisor)<threshold) divisor=threshold;
    }
    const REAL pivot=values[size_t(pivot_position)*step+copy_base];
    for(unsigned i=threadIdx.x>>copy_bits;i<count;i+=row_slots) {
        REAL factor=values[size_t(lower+i)*step+copy_base];
        if constexpr(!Normalized) factor/=divisor;
        if(factor!=REAL(0)) {
            REAL* target=&values[size_t(destinations[base+i])*step+copy_base];
            atomicAdd(target,-factor*pivot);
        }
    }
}

// Both execution granularities consume the same ordered destination groups.
template<bool Packed>
__device__ inline void apply_update_group(REAL* values,unsigned stride,unsigned copy,
        const UpdateGroup* groups,const uint64_t* records,const NormalizedUpdatePair* pairs,
        unsigned index,bool inline_singletons) {
    unsigned destination,first,last,lower=0,pivot=0;
    bool single;
    if constexpr(Packed) {
        const uint64_t group=records[index];
        destination=unsigned(group&packed_update_position_mask);
        single=(group&packed_update_single_bit)!=0;
        if(single) {
            lower=unsigned((group>>21)&packed_update_position_mask);
            pivot=unsigned((group>>42)&packed_update_position_mask);
            first=last=0;
        } else {
            first=unsigned((group>>21)&packed_update_pair_mask);
            last=first+unsigned(group>>48);
        }
    } else {
        const auto group=groups[index];
        single=inline_singletons && (group.destination&inline_update_pair_bit);
        destination=single ? group.destination&~inline_update_pair_bit : group.destination;
        first=group.begin;last=group.end;
        if(single) {lower=group.begin;pivot=group.end;}
    }
    REAL acc=values[size_t(destination)*stride+copy];
    if(single) {
        const REAL factor=values[size_t(lower)*stride+copy];
        if(factor!=REAL(0))acc-=factor*values[size_t(pivot)*stride+copy];
    } else for(unsigned p=first;p<last;++p) {
        const auto pair=pairs[p];const REAL factor=values[size_t(pair.lower)*stride+copy];
        if(factor!=REAL(0))acc-=factor*values[size_t(pair.pivot)*stride+copy];
    }
    values[size_t(destination)*stride+copy]=acc;
}

// Each destination/copy has one writer; preserve zero-factor skipping.
__global__ void mapped_group_update(REAL* values,unsigned copies,unsigned stride,unsigned copy_bits,
        const UpdateGroup* groups,const NormalizedUpdatePair* pairs,unsigned begin,unsigned end,bool inline_singletons) {
    const unsigned lanes=1u<<copy_bits,rows=blockDim.x>>copy_bits;
    const unsigned copy=blockIdx.y*lanes+(threadIdx.x&(lanes-1));
    const size_t index=size_t(begin)+size_t(blockIdx.x)*rows+(threadIdx.x>>copy_bits);
    if(copy>=copies || index>=end)return;
    apply_update_group<false>(values,stride,copy,groups,nullptr,pairs,unsigned(index),inline_singletons);
}

__global__ void packed_group_update(REAL* values,unsigned copies,unsigned stride,unsigned copy_bits,
        const uint64_t* groups,const NormalizedUpdatePair* pairs,unsigned begin,unsigned end) {
    const unsigned lanes=1u<<copy_bits,rows=blockDim.x>>copy_bits;
    const unsigned copy=blockIdx.y*lanes+(threadIdx.x&(lanes-1));
    const size_t index=size_t(begin)+size_t(blockIdx.x)*rows+(threadIdx.x>>copy_bits);
    if(copy>=copies || index>=end)return;
    apply_update_group<true>(values,stride,copy,nullptr,groups,pairs,unsigned(index),false);
}

struct FusedNormalization { unsigned position,diagonal; };
struct FusedStep { UpdateRange columns,immediate,deferred,strips,normalization; };

// One matrix per block, independent of template reuse. Lanes own independent
// update targets, and factors stay in their ordinary AoS allocation.
template<bool Perturb,bool Packed>
__device__ void fused_panel_compute(REAL* values,const unsigned* diag,
        const int* order,const FusedStep* steps,unsigned count,
        const UpdateGroup* groups,const uint64_t* records,const NormalizedUpdatePair* pairs,
        const DeferredUpdateStrip* strips,const unsigned* destinations,const FusedNormalization* normalizations,
        bool inline_singletons,REAL threshold) {
    const unsigned lane=threadIdx.x%GLU_GROUP_SIZE,group=threadIdx.x/GLU_GROUP_SIZE;
    for(unsigned level=0;level<count;++level) {
        const auto step=steps[level];
        if constexpr(Perturb) {
            for(unsigned p=step.columns.begin+threadIdx.x;p<step.columns.end;p+=blockDim.x) {
                const unsigned diagonal=diag[unsigned(order[p])];
                if(fabs(values[diagonal])<threshold)values[diagonal]=threshold;
            }
            __syncthreads();
        }
        for(unsigned p=step.normalization.begin+threadIdx.x;p<step.normalization.end;p+=blockDim.x) {
            const auto entry=normalizations[p];values[entry.position]/=values[entry.diagonal];
        }
        __syncthreads();
        for(size_t p=size_t(step.immediate.begin)+threadIdx.x;p<step.immediate.end;p+=blockDim.x)
            apply_update_group<Packed>(values,1,0,groups,records,pairs,unsigned(p),inline_singletons);
        __syncthreads();
        if(step.deferred.begin!=step.deferred.end || step.strips.begin!=step.strips.end) {
            for(size_t p=size_t(step.deferred.begin)+threadIdx.x;p<step.deferred.end;p+=blockDim.x)
                apply_update_group<Packed>(values,1,0,groups,records,pairs,unsigned(p),inline_singletons);
            for(unsigned p=step.strips.begin+group;p<step.strips.end;p+=blockDim.x/GLU_GROUP_SIZE) {
                const auto strip=strips[p];
                if(lane<strip.count) {
                    const unsigned destination=destinations[strip.destinations+lane];
                    REAL acc=values[destination];
                    // Expose independent operands while keeping contribution order.
                    unsigned q=0;
                    for(;q+1<strip.length;q+=2) {
                        const auto first=pairs[strip.pairs+q*strip.count+lane];
                        const auto second=pairs[strip.pairs+(q+1)*strip.count+lane];
                        const REAL lower0=values[first.lower],lower1=values[second.lower];
                        const REAL pivot0=values[first.pivot],pivot1=values[second.pivot];
                        if(lower0!=REAL(0))acc-=lower0*pivot0;
                        if(lower1!=REAL(0))acc-=lower1*pivot1;
                    }
                    for(;q<strip.length;++q) {
                        const auto pair=pairs[strip.pairs+q*strip.count+lane];
                        const REAL factor=values[pair.lower];
                        if(factor!=REAL(0))acc-=factor*values[pair.pivot];
                    }
                    values[destination]=acc;
                }
            }
            __syncthreads();
        }
    }
}

template<bool Perturb,bool Packed>
__global__ void fused_panel_factorize(REAL* values,unsigned nnz,const unsigned* diag,
        const int* order,const FusedStep* steps,unsigned count,
        const UpdateGroup* groups,const uint64_t* records,const NormalizedUpdatePair* pairs,
        const DeferredUpdateStrip* strips,const unsigned* destinations,const FusedNormalization* normalizations,
        bool inline_singletons,REAL threshold) {
    fused_panel_compute<Perturb,Packed>(values+size_t(blockIdx.x)*nnz,diag,order,steps,
        count,groups,records,pairs,strips,destinations,normalizations,inline_singletons,threshold);
}

struct FusedTaskDescriptor {
    size_t factor_offset,compact_offset;
    unsigned nnz,compact_count,levels;
    const unsigned *diag,*positions;
    const int* order;
    const FusedStep* steps;
    const UpdateGroup* groups;
    const uint64_t* records;
    const NormalizedUpdatePair* pairs;
    const DeferredUpdateStrip* strips;
    const unsigned* destinations;
    const FusedNormalization* normalizations;
    bool inline_singletons,perturb;
    REAL threshold;
};

__global__ void fused_task_set(REAL* factors,const REAL* input,
                              const FusedTaskDescriptor* tasks,bool compact) {
    const auto task=tasks[blockIdx.x];
    REAL* values=factors+task.factor_offset;
    if(compact) {
        for(size_t p=threadIdx.x;p<task.nnz;p+=blockDim.x)values[p]=REAL(0);
        __syncthreads();
        for(size_t p=threadIdx.x;p<task.compact_count;p+=blockDim.x)
            values[task.positions[p]]=input[task.compact_offset+p];
        __syncthreads();
    }
    if(task.perturb) {
        if(task.records)fused_panel_compute<true,true>(values,task.diag,task.order,task.steps,
            task.levels,task.groups,task.records,task.pairs,task.strips,task.destinations,task.normalizations,task.inline_singletons,task.threshold);
        else fused_panel_compute<true,false>(values,task.diag,task.order,task.steps,
            task.levels,task.groups,task.records,task.pairs,task.strips,task.destinations,task.normalizations,task.inline_singletons,task.threshold);
    } else {
        if(task.records)fused_panel_compute<false,true>(values,task.diag,task.order,task.steps,
            task.levels,task.groups,task.records,task.pairs,task.strips,task.destinations,task.normalizations,task.inline_singletons,task.threshold);
        else fused_panel_compute<false,false>(values,task.diag,task.order,task.steps,
            task.levels,task.groups,task.records,task.pairs,task.strips,task.destinations,task.normalizations,task.inline_singletons,task.threshold);
    }
}

#include "update-tiles.cuh"

template<bool Perturb = false>
__global__ void mapped_normalize_level_soa(REAL* values,unsigned copies,
        const unsigned* col,const unsigned* diag,const int* order,int begin,
        unsigned copy_bits,REAL threshold=0,unsigned value_stride=0)
{
    if(!value_stride) value_stride=copies;
    __shared__ REAL divisors[32];
    const unsigned source=unsigned(order[begin+blockIdx.x]);
    const unsigned diagonal=diag[source],lower=diagonal+1,count=col[source+1]-lower;
    const unsigned lanes=1u<<copy_bits,slot=threadIdx.x>>copy_bits;
    const unsigned lane=threadIdx.x&(lanes-1),copy=blockIdx.y*lanes+lane;
    if(slot==0 && copy<copies) {
        REAL divisor=values[size_t(diagonal)*value_stride+copy];
        if constexpr(Perturb) if(fabs(divisor)<threshold) {
            divisor=threshold;
            values[size_t(diagonal)*value_stride+copy]=divisor;
        }
        divisors[lane]=divisor;
    }
    __syncthreads();
    if(copy>=copies) return;
    const REAL divisor=divisors[lane];
    for(unsigned i=slot;i<count;i+=blockDim.x>>copy_bits)
        values[size_t(lower+i)*value_stride+copy]/=divisor;
}

// Ordinary tiled matrix transpose. Linear tile grid avoids a new grid.y limit
// for a large symbolic payload. Both input and output include partial tiles.
__global__ void transpose_task_values(const REAL* input,REAL* output,unsigned rows,unsigned columns,
        unsigned input_stride=0,unsigned output_stride=0) {
    if(!input_stride) input_stride=columns;
    if(!output_stride) output_stride=rows;
    __shared__ REAL tile[32][33];
    const unsigned tiles=columns/32+(columns%32!=0);
    const unsigned tx=blockIdx.x%tiles,ty=blockIdx.x/tiles;
    const unsigned x=tx*32+threadIdx.x,y=ty*32+threadIdx.y;
    for(unsigned j=0;j<32;j+=8)
        if(x<columns && y+j<rows) tile[threadIdx.y+j][threadIdx.x]=input[size_t(y+j)*input_stride+x];
    __syncthreads();
    const unsigned out_col=ty*32+threadIdx.x,out_row=tx*32+threadIdx.y;
    for(unsigned j=0;j<32;j+=8)
        if(out_col<rows && out_row+j<columns)
            output[size_t(out_row+j)*output_stride+out_col]=tile[threadIdx.x][threadIdx.y+j];
}

__global__ void RL_onecol_cleartmpMem(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const size_t scratch_stride = 0)
{
    tmpMem += size_t(blockIdx.y) * scratch_stride;
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    unsigned offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];
            tmpMem[size_t(stream) * n + ridx]= 0;
        }
        offset += blockDim.x;
    }
}


namespace {
// Resolve once per task, including its actual tail-batch size. Keep execution
// and diagnostics on the same effective policy; Auto never adds device residency.
SnapshotPolicy resolve_snapshot_policy(SnapshotPolicy requested,UpdateMode mode,unsigned copies) {
    if(requested!=SnapshotPolicy::Auto)return requested;
    return copies>1 && mode==UpdateMode::Panel
        ? SnapshotPolicy::Shared : SnapshotPolicy::PerCopy;
}
// Copy lanes fill naturally up to the fixed kernel cap. This is also the
// physical row alignment, so tiny batches do not pay a 32-column allocation.
unsigned filled_copy_bits(unsigned copies,unsigned cap) {
    unsigned bits=0;
    while(bits<cap && (1u<<bits)<copies) ++bits;
    return bits;
}
// CUDA FP64 normalizes small SoA levels before their updates. Other backend /
// precision combinations defer normalization; the single-copy mapped AoS path always
// defers it. This policy does not change the task's input/output contract.
#if !defined(GLU_USE_HIP) && !defined(GLU_USE_FP32)
constexpr bool normalize_soa_levels=true;
#else
constexpr bool normalize_soa_levels=false;
#endif
void gpu_check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

size_t checked_bytes(uint64_t count, size_t element_size) {
    if (count > std::numeric_limits<size_t>::max()/element_size)
        throw std::overflow_error("GPU allocation size overflow");
    return size_t(count)*element_size;
}

size_t checked_sum(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max()-a)
        throw std::overflow_error("GPU allocation total overflow");
    return a+b;
}

const std::vector<unsigned>& require_compact_order(const std::vector<unsigned>& positions) {
    if(positions.empty())throw std::invalid_argument("compact input order must not be empty");
    return positions;
}

template<class T, bool Pinned = false> struct Buffer {
    T* ptr = nullptr;
    size_t bytes = 0;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (ptr) { if (Pinned) cudaFreeHost(ptr); else cudaFree(ptr); } }
    void allocate(size_t count, size_t copies = 1) {
        if (!copies || count > std::numeric_limits<size_t>::max() / sizeof(T) / copies)
            throw std::overflow_error("GPU allocation size overflow");
        if (!count) return;
        if (Pinned) gpu_check(cudaMallocHost(reinterpret_cast<void**>(&ptr), count*copies*sizeof(T)), "allocate pinned buffer");
        else gpu_check(cudaMalloc(reinterpret_cast<void**>(&ptr), count*copies*sizeof(T)), "allocate device buffer");
        bytes = count*copies*sizeof(T);
    }
    void upload(const std::vector<T>& input) {
        allocate(input.size());
        if (!input.empty())
            gpu_check(cudaMemcpy(ptr, input.data(), bytes, cudaMemcpyHostToDevice), "upload structure");
    }
};
}

// Separate destinations let all copies factor independently. For Shared, the
// first uploaded copy is read-only until this stream finishes the broadcast.
__global__ void restore_snapshot(REAL* destination,const REAL* source,unsigned nnz) {
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<nnz) destination[size_t(blockIdx.y)*nnz+at]=source[at];
}

__global__ void scatter_compact_input(REAL* values,const REAL* compact,
                                     const unsigned* positions,unsigned count,unsigned nnz) {
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<count)values[size_t(blockIdx.y)*nnz+positions[at]]=compact[size_t(blockIdx.y)*count+at];
}

// Fuse the input transpose with scatter instead of materializing full AoS first.
// A shared tile gives contiguous compact-entry loads and SoA-copy stores.
// Positions need not be ordered. Unwritten factor slots and padding are cleared
// before this launch; each listed position is unique across the entire input.
__global__ void scatter_compact_input_soa(REAL* values,const REAL* compact,
                                         const unsigned* positions,unsigned count,
                                         unsigned copies,unsigned stride) {
    __shared__ REAL tile[32][33];
    const unsigned entry_base=blockIdx.x*32,copy_base=blockIdx.y*32;
    const unsigned entry=entry_base+threadIdx.x,copy=copy_base+threadIdx.y;
    for(unsigned offset=0;offset<32;offset+=8)
        if(entry<count && copy+offset<copies)
            tile[threadIdx.y+offset][threadIdx.x]=compact[size_t(copy+offset)*count+entry];
    __syncthreads();
    const unsigned output_copy=copy_base+threadIdx.x;
    for(unsigned offset=0;offset<32;offset+=8) {
        const unsigned input_entry=entry_base+threadIdx.y+offset;
        if(input_entry<count && output_copy<copies)
            values[size_t(positions[input_entry])*stride+output_copy]=tile[threadIdx.x][threadIdx.y+offset];
    }
}

struct GpuMatrix::Impl {
    unsigned n, nnz, slots, max_batch, max_grid_x;
    unsigned fused_block_threads=256;
    unsigned deferred_columns_count=0;
    bool perturb;
    UpdateMode update_mode;
    SnapshotPolicy snapshot_policy;
    size_t shared_bytes = 0, map_bytes = 0, plan_bytes = 0;
    double map_setup_ms = 0, plan_setup_ms = 0;
    double snapshot_setup_ms=0;
    double compact_setup_ms=0;
    unsigned compact_elements=0;
    bool panel_mode=false,native_mmac=false;
    REAL threshold = 0;
    std::string device;
    std::vector<int> levels, columns;
    std::vector<unsigned> csr_rows, csr_diags, diagonals;
    Buffer<unsigned> col, row, diag, csr_row, csr_col, csr_diag;
    Buffer<int> level_idx, deferred_columns;
    Buffer<uint64_t> level_targets;
    std::vector<uint64_t> target_prefix;
    Buffer<uint64_t> map_bases;
    Buffer<unsigned> map_destinations, map_pivots;
    std::vector<UpdateRange> panel_immediate,panel_deferred,panel_dense;
    bool inline_singletons=false;
    Buffer<UpdateGroup> panel_groups;
    Buffer<uint64_t> panel_records;
    Buffer<NormalizedUpdatePair> panel_pairs;
    Buffer<FusedStep> fused_steps;
    Buffer<DeferredUpdateStrip> fused_strips;
    Buffer<unsigned> fused_destinations;
    Buffer<FusedNormalization> fused_normalizations;
    Buffer<DenseUpdateTile> dense_tiles;
    Buffer<unsigned> dense_indices;
    Buffer<REAL, true> input;
    Buffer<REAL> initial_values;
    Buffer<unsigned> input_positions;

    unsigned value_pitch(unsigned copies) const {
        if(!copies || copies>max_batch) throw std::invalid_argument("batch size exceeds device grid limit");
        if(update_mode==UpdateMode::Scan || update_mode==UpdateMode::Fused || copies==1) return 0;
        const unsigned lanes=1u<<filled_copy_bits(copies,5);
        const uint64_t pitch=(uint64_t(copies)+lanes-1)/lanes*lanes;
        if(pitch>std::numeric_limits<unsigned>::max()) throw std::overflow_error("SoA leading dimension overflow");
        return unsigned(pitch);
    }

    Impl(const Symbolic_Matrix& a, unsigned requested_slots, bool use_perturb, UpdateMode mode,SnapshotPolicy snapshot,
         const std::vector<unsigned>& positions)
        : n(a.n), nnz(a.nnz), slots(0), perturb(use_perturb), update_mode(mode),
          snapshot_policy(snapshot),
          levels(a.level_ptr), columns(a.level_idx), csr_rows(a.csr_r_ptr),
          csr_diags(a.csr_diag_ptr), diagonals(a.l_col_ptr)
    {
        if (!n || !nnz || !requested_slots || levels.size() < 2)
            throw std::invalid_argument("empty matrix or zero scratch slots");
        if (mode != UpdateMode::Scan && mode != UpdateMode::Mapped && mode != UpdateMode::Panel && mode != UpdateMode::PanelTiled && mode != UpdateMode::Fused)
            throw std::invalid_argument("invalid update mode");
        if(snapshot!=SnapshotPolicy::Auto && snapshot!=SnapshotPolicy::PerCopy &&
           snapshot!=SnapshotPolicy::Shared && snapshot!=SnapshotPolicy::Device)
            throw std::invalid_argument("invalid snapshot policy");
        if(!positions.empty()) {
            const auto start=std::chrono::steady_clock::now();
            if(positions.size()>nnz)throw std::invalid_argument("too many compact input positions");
            std::vector<unsigned char> seen(nnz,0);
            for(unsigned p:positions) {
                if(p>=nnz || seen[p])throw std::invalid_argument("compact input positions must be unique factor slots");
                seen[p]=1;
            }
            compact_elements=unsigned(positions.size());
            compact_setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        panel_mode=mode==UpdateMode::Panel || mode==UpdateMode::PanelTiled || mode==UpdateMode::Fused;
        GpuMatrix::select_device();
        cudaDeviceProp prop;
        gpu_check(cudaGetDeviceProperties(&prop, 0), "get device properties");
        if ((prop.warpSize != 32 && prop.warpSize != 64) || prop.maxThreadsPerBlock < 1024)
            throw std::runtime_error("requires warp32/wave64 and 1024-thread blocks");
        device = prop.name;
        max_batch = unsigned(prop.maxGridSize[1]);
        max_grid_x = unsigned(prop.maxGridSize[0]);
#if defined(GLU_USE_HIP)
        native_mmac=mode==UpdateMode::PanelTiled && std::string(prop.gcnArchName).rfind("gfx936",0)==0;
#endif
        unsigned needed = 1;
        for (size_t i=1; i<levels.size(); ++i) {
            const unsigned width = unsigned(levels[i]-levels[i-1]);
            if (mode != UpdateMode::Scan && width > unsigned(prop.maxGridSize[0]))
                throw std::invalid_argument("mapped level width exceeds device grid limit");
            // Small levels execute serially in slot zero.
            if (mode == UpdateMode::Scan && width > 16) needed = std::max(needed, width);
        }
        slots = mode == UpdateMode::Scan ? std::min(requested_slots, needed) : 0;
        if(mode!=UpdateMode::Fused)col.upload(a.sym_c_ptr);
        diag.upload(a.l_col_ptr);
        if(!panel_mode) { csr_row.upload(a.csr_r_ptr);csr_diag.upload(a.csr_diag_ptr); }
        if(mode==UpdateMode::Scan) { row.upload(a.sym_r_idx);csr_col.upload(a.csr_c_idx); }
        level_idx.upload(a.level_idx);
        if(panel_mode) {
            const auto begin=std::chrono::steady_clock::now();
            inline_singletons=a.nnz<inline_update_pair_bit;
            auto plan=make_panel_update_plan(a,32,mode==UpdateMode::PanelTiled,inline_singletons);
            // Target parallelism guides this template's block size. A set
            // uses the widest participating template, preserving one launch.
            if(mode==UpdateMode::Fused && plan.groups.size()>uint64_t(a.num_lev)*1024)fused_block_threads=1024;
            if(mode!=UpdateMode::Fused)sort_deferred_groups_locally(plan,inline_singletons);
            DeferredStripPlan strip_plan;
            if(mode==UpdateMode::Fused)strip_plan=organize_fused_updates(plan,inline_singletons);
            auto records=pack_update_groups(plan,nnz,inline_singletons);
            std::vector<FusedStep> steps;
            std::vector<FusedNormalization> normalizations;
            if(mode==UpdateMode::Fused) {
                for(unsigned lev=0;lev<unsigned(a.num_lev);++lev) {
                    const unsigned begin=unsigned(normalizations.size());
                    for(int p=a.level_ptr[lev];p<a.level_ptr[lev+1];++p) {
                        const unsigned column=unsigned(a.level_idx[p]),diagonal=a.l_col_ptr[column];
                        for(unsigned position=diagonal+1;position<a.sym_c_ptr[column+1];++position)
                            normalizations.push_back({position,diagonal});
                    }
                    steps.push_back({{unsigned(a.level_ptr[lev]),unsigned(a.level_ptr[lev+1])},plan.immediate[lev],plan.deferred[lev],strip_plan.ranges[lev],{begin,unsigned(normalizations.size())}});
                }
            }
            plan_bytes=records.empty() ? checked_bytes(plan.groups.size(),sizeof(UpdateGroup)) : checked_bytes(records.size(),sizeof(uint64_t));
            for(size_t bytes:{checked_bytes(plan.pairs.size(),sizeof(NormalizedUpdatePair)),
                              checked_bytes(plan.dense_tiles.size(),sizeof(DenseUpdateTile)),checked_bytes(plan.dense_indices.size(),sizeof(unsigned))})
                plan_bytes=checked_sum(plan_bytes,bytes);
            plan_bytes=checked_sum(plan_bytes,checked_bytes(steps.size(),sizeof(FusedStep)));
            plan_bytes=checked_sum(plan_bytes,checked_bytes(strip_plan.strips.size(),sizeof(DeferredUpdateStrip)));
            plan_bytes=checked_sum(plan_bytes,checked_bytes(strip_plan.destinations.size(),sizeof(unsigned)));
            plan_bytes=checked_sum(plan_bytes,checked_bytes(normalizations.size(),sizeof(FusedNormalization)));
            if(plan_bytes>GpuMatrix::free_device_bytes())throw std::runtime_error("panel plan exceeds free device memory");
            for(const auto range:plan.immediate)if(range.end-range.begin>max_grid_x)throw std::invalid_argument("panel group grid exceeds device limit");
            for(const auto range:plan.deferred)if(range.end-range.begin>max_grid_x)throw std::invalid_argument("panel group grid exceeds device limit");
            for(const auto range:plan.dense_ranges)if(range.end-range.begin>max_grid_x)throw std::invalid_argument("dense tile grid exceeds device limit");
            panel_immediate=std::move(plan.immediate);panel_deferred=std::move(plan.deferred);panel_dense=std::move(plan.dense_ranges);
            if(records.empty())panel_groups.upload(plan.groups);else panel_records.upload(records);
            panel_pairs.upload(plan.pairs);
            dense_tiles.upload(plan.dense_tiles);dense_indices.upload(plan.dense_indices);
            fused_steps.upload(steps);
            fused_strips.upload(strip_plan.strips);fused_destinations.upload(strip_plan.destinations);
            fused_normalizations.upload(normalizations);
            plan_setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        }
        if (mode == UpdateMode::Mapped) {
            const auto begin = std::chrono::steady_clock::now();
            build_update_map(a);
            target_prefix.push_back(0);
            for(int column : columns) {
                const uint64_t targets=a.sym_c_ptr[column+1]>a.l_col_ptr[column]+1
                    ? uint64_t(csr_rows[column+1]-csr_diags[column]-1) : 0;
                if(targets>std::numeric_limits<uint64_t>::max()-target_prefix.back())
                    throw std::overflow_error("level target count overflow");
                target_prefix.push_back(target_prefix.back()+targets);
            }
            std::vector<int> deferred;
            for(size_t lev=0;lev+1<levels.size();++lev)
                if(levels[lev+1]-levels[lev]<=16) {
                    if(target_prefix[levels[lev+1]]-target_prefix[levels[lev]]>max_grid_x)
                        throw std::invalid_argument("mapped level targets exceed device grid limit");
                    deferred.insert(deferred.end(),columns.begin()+levels[lev],columns.begin()+levels[lev+1]);
                }
            if(deferred.size()>max_grid_x) throw std::invalid_argument("deferred columns exceed device grid limit");
            deferred_columns_count=unsigned(deferred.size());
            level_targets.upload(target_prefix);
            deferred_columns.upload(deferred);
            map_setup_ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        }
        for (size_t bytes : {col.bytes, row.bytes, diag.bytes, csr_row.bytes,
                             csr_col.bytes, csr_diag.bytes, level_idx.bytes, map_bytes, level_targets.bytes, deferred_columns.bytes,plan_bytes})
            shared_bytes = checked_sum(shared_bytes, bytes);
        input.allocate(nnz);
        std::copy(a.val.begin(), a.val.end(), input.ptr);
        if(!positions.empty()) {
            const auto start=std::chrono::steady_clock::now();
            input_positions.upload(positions);
            shared_bytes=checked_sum(shared_bytes,input_positions.bytes);
            compact_setup_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        if(snapshot_policy==SnapshotPolicy::Device) {
            const auto start=std::chrono::steady_clock::now();
            initial_values.upload(a.val);
            shared_bytes=checked_sum(shared_bytes,initial_values.bytes);
            snapshot_setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        if (perturb) {
            for (unsigned j=0; j<n; ++j) {
                REAL sum = 0;
                for (unsigned p=a.sym_c_ptr[j]; p<a.sym_c_ptr[j+1]; ++p) sum += std::abs(a.val[p]);
                threshold = std::max(threshold, sum);
            }
            threshold *= REAL(3.45e-4f);
        }
    }

    void build_update_map(const Symbolic_Matrix& a) {
        // Count before allocating the potentially much larger-than-factor map.
        uint64_t total = 0;
        for (unsigned c=0; c<n; ++c) {
            const uint64_t lower = a.sym_c_ptr[c+1]-a.l_col_ptr[c]-1;
            const uint64_t targets = a.csr_r_ptr[c+1]-a.csr_diag_ptr[c]-1;
            if (targets && lower > std::numeric_limits<uint64_t>::max()/targets)
                throw std::overflow_error("update map size overflow");
            const uint64_t added = lower*targets;
            if (added > std::numeric_limits<uint64_t>::max()-total)
                throw std::overflow_error("update map size overflow");
            total += added;
        }
        map_bytes = checked_sum(checked_bytes(uint64_t(n)+1, sizeof(uint64_t)),
                    checked_sum(checked_bytes(total, sizeof(unsigned)),
                                checked_bytes(a.csr_c_idx.size(), sizeof(unsigned))));
        const size_t available = GpuMatrix::free_device_bytes();
        if (map_bytes > available)
            throw std::runtime_error("mapped updates require " + std::to_string(map_bytes) +
                " additional device bytes, but only " + std::to_string(available) +
                " are free; use UpdateMode::Scan to avoid the map allocation");
        std::vector<uint64_t> bases(size_t(n)+1);
        std::vector<unsigned> destinations;
        destinations.reserve(size_t(total));
        std::vector<unsigned> pivots(a.csr_c_idx.size());
        for (unsigned c=0; c<n; ++c) {
            bases[c] = destinations.size();
            for (unsigned q=a.csr_diag_ptr[c]+1; q<a.csr_r_ptr[c+1]; ++q) {
                const unsigned target = a.csr_c_idx[q];
                unsigned at = a.sym_c_ptr[target];
                const unsigned end = a.sym_c_ptr[target+1];
                while (at<end && a.sym_r_idx[at]<c) ++at;
                if (at==end || a.sym_r_idx[at]!=c)
                    throw std::runtime_error("update map missing pivot");
                pivots[q] = at++;
                for (unsigned source=a.l_col_ptr[c]+1; source<a.sym_c_ptr[c+1]; ++source) {
                    const unsigned row = a.sym_r_idx[source];
                    while (at<end && a.sym_r_idx[at]<row) ++at;
                    // Mapped mode requires a structurally closed update pattern.
                    if (at==end || a.sym_r_idx[at]!=row)
                        throw std::runtime_error("update map missing lower row");
                    destinations.push_back(at);
                }
            }
        }
        bases[n] = destinations.size();
        if (destinations.size()!=total) throw std::logic_error("update map size mismatch");
        map_bases.upload(bases); map_destinations.upload(destinations); map_pivots.upload(pivots);
    }
};

struct GpuTask::Impl {
    std::shared_ptr<const GpuMatrix::Impl> matrix;
    const SnapshotPolicy snapshot_policy;
    Buffer<REAL> values, scratch, interleaved_values,compact_values;
    Buffer<REAL, true> output;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    unsigned copies,copy_bits,soa_stride;
    bool pending=false,ready=false,failed=false,input_queued=false;
    unsigned copy_tiles(unsigned bits) const { return (copies+(1u<<bits)-1)>>bits; }
    explicit Impl(std::shared_ptr<const GpuMatrix::Impl> m,unsigned batch)
        : matrix(std::move(m)),snapshot_policy(resolve_snapshot_policy(matrix->snapshot_policy,matrix->update_mode,batch)),
          copies(batch),copy_bits(filled_copy_bits(batch,5)),soa_stride(matrix->value_pitch(batch)) {
        if(soa_stride && !matrix->panel_mode)for(size_t lev=0;lev+1<matrix->levels.size();++lev)
            if(matrix->target_prefix[matrix->levels[lev+1]]-matrix->target_prefix[matrix->levels[lev]]>matrix->max_grid_x)
                throw std::invalid_argument("mapped level targets exceed device grid limit");
        values.allocate(matrix->nnz,copies);
        if(matrix->compact_elements)compact_values.allocate(matrix->compact_elements,copies);
        if(soa_stride) {
            const uint64_t tiles=uint64_t(matrix->nnz/32+(matrix->nnz%32!=0))*(copies/32+(copies%32!=0));
            if(tiles>matrix->max_grid_x) throw std::invalid_argument("SoA transpose grid exceeds device limit");
            interleaved_values.allocate(matrix->nnz,soa_stride);
        }
        if(matrix->update_mode==UpdateMode::Scan) scratch.allocate(size_t(matrix->n)*matrix->slots,copies);
        output.allocate(matrix->nnz,copies);
        gpu_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create task stream");
        try {
            if(matrix->update_mode==UpdateMode::Scan) {
                gpu_check(cudaMemsetAsync(scratch.ptr,0,scratch.bytes,stream),"initialize scratch");
                gpu_check(cudaStreamSynchronize(stream),"finish task preparation");
            }
        } catch(...) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); throw; }
    }
    void enqueue_aos_input(const REAL* input) {
        if(input) {
            gpu_check(cudaMemcpyAsync(values.ptr,input,values.bytes,cudaMemcpyHostToDevice,stream),"upload matrix input");
            return;
        }
        if(snapshot_policy==SnapshotPolicy::Device) {
            restore_snapshot<<<dim3(matrix->nnz/256u+(matrix->nnz%256u!=0),copies),256,0,stream>>>(
                values.ptr,matrix->initial_values.ptr,matrix->nnz);
            gpu_check(cudaGetLastError(),"restore device snapshot");
            return;
        }
        if(snapshot_policy==SnapshotPolicy::Shared && copies>1) {
            gpu_check(cudaMemcpyAsync(values.ptr,matrix->input.ptr,matrix->input.bytes,cudaMemcpyHostToDevice,stream),"upload shared snapshot");
            restore_snapshot<<<dim3(matrix->nnz/256u+(matrix->nnz%256u!=0),copies-1),256,0,stream>>>(
                values.ptr+matrix->nnz,values.ptr,matrix->nnz);
            gpu_check(cudaGetLastError(),"broadcast shared snapshot");
            return;
        }
        for(unsigned copy=0;copy<copies;++copy)
            gpu_check(cudaMemcpyAsync(values.ptr+size_t(copy)*matrix->nnz,matrix->input.ptr,
                size_t(matrix->nnz)*sizeof(REAL),cudaMemcpyHostToDevice,stream),"upload matrix input");
    }
    void enqueue_input(const REAL* input=nullptr) {
        enqueue_aos_input(input);
        // Every input path prepares the execution layout before numeric work.
        // This boundary is independent of compact support and Graph capture.
        if(soa_stride)transpose(true);
    }
    void enqueue_compact(const REAL* input) {
        gpu_check(cudaMemcpyAsync(compact_values.ptr,input,compact_values.bytes,cudaMemcpyHostToDevice,stream),"upload compact matrix input");
        // Clear the whole execution allocation, including SoA padding: omitted
        // input positions must not inherit fill values from a previous LU.
        if(soa_stride) {
            gpu_check(cudaMemsetAsync(interleaved_values.ptr,0,interleaved_values.bytes,stream),"clear SoA factor slots for compact input");
            scatter_compact_input_soa<<<dim3(matrix->compact_elements/32u+(matrix->compact_elements%32u!=0),copies/32u+(copies%32u!=0)),dim3(32,8),0,stream>>>(
                interleaved_values.ptr,compact_values.ptr,matrix->input_positions.ptr,matrix->compact_elements,copies,soa_stride);
        } else {
            gpu_check(cudaMemsetAsync(values.ptr,0,values.bytes,stream),"clear factor slots for compact input");
            scatter_compact_input<<<dim3(matrix->compact_elements/256u+(matrix->compact_elements%256u!=0),copies),256,0,stream>>>(
                values.ptr,compact_values.ptr,matrix->input_positions.ptr,matrix->compact_elements,matrix->nnz);
        }
        gpu_check(cudaGetLastError(),"scatter compact matrix input");
    }
    void transpose(bool to_interleaved) {
        const unsigned rows=to_interleaved ? copies : matrix->nnz,columns=to_interleaved ? matrix->nnz : copies;
        const unsigned blocks=unsigned(uint64_t(rows/32+(rows%32!=0))*(columns/32+(columns%32!=0)));
        transpose_task_values<<<blocks,dim3(32,8),0,stream>>>(to_interleaved ? values.ptr : interleaved_values.ptr,
            to_interleaved ? interleaved_values.ptr : values.ptr,rows,columns,
            to_interleaved ? matrix->nnz : soa_stride,to_interleaved ? soa_stride : matrix->nnz);
        gpu_check(cudaGetLastError(),"transpose task values");
    }
    template<UpdateMode Mode,bool Perturb> void enqueue_numeric() {
        auto& t=*this;
        const auto& a=*matrix;
        const unsigned column_threads=copies>=16 ? 256 : 1024;
        const bool soa=soa_stride!=0;
        if(a.update_mode==UpdateMode::Fused) {
            if(a.panel_records.ptr)
                fused_panel_factorize<Perturb,true><<<copies,a.fused_block_threads,0,stream>>>(values.ptr,a.nnz,a.diag.ptr,a.level_idx.ptr,
                    a.fused_steps.ptr,unsigned(a.levels.size()-1),nullptr,a.panel_records.ptr,a.panel_pairs.ptr,a.fused_strips.ptr,a.fused_destinations.ptr,a.fused_normalizations.ptr,a.inline_singletons,a.threshold);
            else
                fused_panel_factorize<Perturb,false><<<copies,a.fused_block_threads,0,stream>>>(values.ptr,a.nnz,a.diag.ptr,a.level_idx.ptr,
                    a.fused_steps.ptr,unsigned(a.levels.size()-1),a.panel_groups.ptr,nullptr,a.panel_pairs.ptr,a.fused_strips.ptr,a.fused_destinations.ptr,a.fused_normalizations.ptr,a.inline_singletons,a.threshold);
            gpu_check(cudaGetLastError(),"factor fused matrix blocks");
            return;
        }
        if constexpr(Mode==UpdateMode::Mapped)if(a.panel_mode) {
            REAL* data=soa ? interleaved_values.ptr : values.ptr;
            const unsigned bits=soa ? copy_bits : 0,stride=soa ? soa_stride : 1,rows=256u>>bits;
            auto update=[&](UpdateRange range) {
                if(range.end>range.begin) {
                    if(a.panel_records.ptr)
                        packed_group_update<<<dim3((range.end-range.begin+rows-1)/rows,copy_tiles(bits)),256,0,stream>>>(
                            data,copies,stride,bits,a.panel_records.ptr,a.panel_pairs.ptr,range.begin,range.end);
                    else
                        mapped_group_update<<<dim3((range.end-range.begin+rows-1)/rows,copy_tiles(bits)),256,0,stream>>>(
                            data,copies,stride,bits,a.panel_groups.ptr,a.panel_pairs.ptr,range.begin,range.end,a.inline_singletons);
                }
            };
            for(size_t lev=0;lev+1<a.levels.size();++lev) {
                const unsigned begin=a.levels[lev],width=a.levels[lev+1]-begin;
                mapped_normalize_level_soa<Perturb><<<dim3(width,copy_tiles(bits)),256,0,stream>>>(
                    data,copies,a.col.ptr,a.diag.ptr,a.level_idx.ptr,begin,bits,a.threshold,stride);
                gpu_check(cudaGetLastError(),"normalize panel sources");
                update(a.panel_immediate[lev]);gpu_check(cudaGetLastError(),"panel immediate update");
                update(a.panel_deferred[lev]);gpu_check(cudaGetLastError(),"panel deferred update");
                const auto range=a.panel_dense[lev];
                if(range.end>range.begin) {
                    panel_tiled_update<<<dim3(range.end-range.begin,(copies+3)/4),256,0,stream>>>(
                        data,copies,stride,a.dense_tiles.ptr,a.dense_indices.ptr,range.begin);
                    gpu_check(cudaGetLastError(),"panel dense update");
                }
            }
            if(soa)transpose(false);
            return;
        }
        for(size_t lev=0;lev+1<a.levels.size();++lev) {
            const int begin=a.levels[lev],end=a.levels[lev+1],width=end-begin;
            if constexpr(Mode==UpdateMode::Mapped) {
                if(width<=16) {
                    if(soa && normalize_soa_levels) {
                        mapped_normalize_level_soa<Perturb><<<dim3(width,copy_tiles(copy_bits)),256,0,stream>>>(
                            interleaved_values.ptr,copies,a.col.ptr,a.diag.ptr,a.level_idx.ptr,begin,copy_bits,a.threshold,soa_stride);
                        gpu_check(cudaGetLastError(),"normalize mapped level");
                    }
                    const uint64_t targets=a.target_prefix[end]-a.target_prefix[begin];
                    if(targets) {
                        if(soa)
                            mapped_level_update_soa<Perturb,normalize_soa_levels><<<dim3(unsigned(targets),copy_tiles(copy_bits)),256,0,stream>>>(
                                interleaved_values.ptr,copies,a.col.ptr,a.diag.ptr,a.csr_diag.ptr,a.level_idx.ptr,a.level_targets.ptr,
                                begin,end,a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr,a.threshold,soa_stride,copy_bits);
                        else
                            mapped_level_update<Perturb><<<dim3(unsigned(targets),copies),256,0,stream>>>(
                                values.ptr,a.nnz,a.col.ptr,a.diag.ptr,a.csr_diag.ptr,a.level_idx.ptr,a.level_targets.ptr,
                                begin,end,a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr,a.threshold);
                        gpu_check(cudaGetLastError(),"update mapped level");
                    }
                    continue;
                }
                if(soa) {
                    // Normalize each source once, then expose its independent
                    // targets as separate blocks in the same copy layout.
                    mapped_normalize_level_soa<Perturb><<<dim3(width,copy_tiles(copy_bits)),256,0,stream>>>(
                        interleaved_values.ptr,copies,a.col.ptr,a.diag.ptr,a.level_idx.ptr,begin,copy_bits,a.threshold,soa_stride);
                    gpu_check(cudaGetLastError(),"normalize wide mapped level");
                    const uint64_t targets=a.target_prefix[end]-a.target_prefix[begin];
                    if(targets) mapped_level_update_soa<Perturb,true><<<dim3(unsigned(targets),copy_tiles(copy_bits)),256,0,stream>>>(
                        interleaved_values.ptr,copies,a.col.ptr,a.diag.ptr,a.csr_diag.ptr,a.level_idx.ptr,a.level_targets.ptr,
                        begin,end,a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr,a.threshold,soa_stride,copy_bits);
                    gpu_check(cudaGetLastError(),"update wide mapped targets");
                    continue;
                }
            }
            if (width > 16) {
                const unsigned default_threads = width > 896 ? 64 : width > 448 ? 128 : 1024;
                // Independent copies supply blocks across grid.y. Avoid making
                // every copy carry a 1024-thread block for a small level.
                const unsigned threads = t.copies >= 16
                    ? std::min(default_threads, 128u) : default_threads;
                for (int offset=0; offset<width;) {
                    // Mapped sources read L directly, so scratch slots impose no chunk limit.
                    const unsigned blocks = Mode == UpdateMode::Mapped ? unsigned(width-offset)
                        : std::min(unsigned(width-offset), a.slots);
                    if (!a.perturb)
                        RL<Mode><<<dim3(blocks,t.copies),threads,0,t.stream>>>(a.col.ptr,a.row.ptr,t.values.ptr,a.diag.ptr,
                            a.csr_row.ptr,a.csr_col.ptr,a.csr_diag.ptr,a.level_idx.ptr,t.scratch.ptr,
                            a.n,a.levels[lev],offset,a.nnz,size_t(a.n)*a.slots,0,
                            a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr);
                    else
                        RL<Mode, true><<<dim3(blocks,t.copies),threads,0,t.stream>>>(a.col.ptr,a.row.ptr,t.values.ptr,a.diag.ptr,
                            a.csr_row.ptr,a.csr_col.ptr,a.csr_diag.ptr,a.level_idx.ptr,t.scratch.ptr,
                            a.n,a.levels[lev],offset,a.nnz,size_t(a.n)*a.slots,a.threshold,
                            a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr);
                    gpu_check(cudaGetLastError(), "launch batched level");
                    offset += int(blocks);
                }
            } else {
                for (int p=a.levels[lev]; p<a.levels[lev+1]; ++p) {
                    const unsigned column = a.columns[p];
                    const unsigned targets = a.csr_rows[column+1]-a.csr_diags[column]-1;
                    if (!a.perturb)
                        RL_onecol_factorizeCurrentCol<false, Mode == UpdateMode::Scan><<<dim3(1,t.copies),column_threads,0,t.stream>>>(a.col.ptr,a.row.ptr,
                            t.values.ptr,a.diag.ptr,column,t.scratch.ptr,0,a.n,a.nnz,size_t(a.n)*a.slots);
                    else
                        RL_onecol_factorizeCurrentCol<true, Mode == UpdateMode::Scan><<<dim3(1,t.copies),column_threads,0,t.stream>>>(a.col.ptr,a.row.ptr,
                            t.values.ptr,a.diag.ptr,column,t.scratch.ptr,0,a.n,a.nnz,size_t(a.n)*a.slots,a.threshold);
                    gpu_check(cudaGetLastError(), "launch column factorization");
                    if (targets) {
                        RL_onecol_updateSubmat<Mode><<<dim3(targets,t.copies),column_threads,0,t.stream>>>(a.col.ptr,a.row.ptr,t.values.ptr,
                            a.csr_col.ptr,a.csr_diag.ptr,column,t.scratch.ptr,0,a.n,a.nnz,size_t(a.n)*a.slots,
                            a.diag.ptr,a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr);
                        gpu_check(cudaGetLastError(), "launch column update");
                    }
                    if constexpr (Mode == UpdateMode::Scan) {
                        RL_onecol_cleartmpMem<<<dim3(1,t.copies),column_threads,0,t.stream>>>(a.col.ptr,a.row.ptr,a.diag.ptr,
                            column,t.scratch.ptr,0,a.n,size_t(a.n)*a.slots);
                        gpu_check(cudaGetLastError(), "launch scratch clear");
                    }
                }
            }
        }
        if(soa) transpose(false);
        if constexpr(Mode==UpdateMode::Mapped) {
            if(a.deferred_columns_count && !(soa && normalize_soa_levels)) {
                // Small-level L remains raw while updates consume L/diagonal.
                // Successors never update it. Normalize once for factor output.
                RL_onecol_factorizeCurrentCol<Perturb,false,true><<<dim3(a.deferred_columns_count,copies),column_threads,0,stream>>>(
                    a.col.ptr,a.row.ptr,values.ptr,a.diag.ptr,0,nullptr,0,a.n,a.nnz,0,a.threshold,a.deferred_columns.ptr,0);
                gpu_check(cudaGetLastError(),"normalize deferred lower columns");
            }
        }
    }
    void enqueue_work(bool graph_output) {
        if(matrix->update_mode!=UpdateMode::Scan) {
            if(matrix->perturb) enqueue_numeric<UpdateMode::Mapped,true>();
            else enqueue_numeric<UpdateMode::Mapped,false>();
        } else {
            if(matrix->perturb) enqueue_numeric<UpdateMode::Scan,true>();
            else enqueue_numeric<UpdateMode::Scan,false>();
        }
        if(graph_output)
            gpu_check(cudaMemcpyAsync(output.ptr,values.ptr,output.bytes,cudaMemcpyDeviceToHost,stream),"download batch factors");
        else for(unsigned copy=0;copy<copies;++copy)
            gpu_check(cudaMemcpyAsync(output.ptr+size_t(copy)*matrix->nnz,values.ptr+size_t(copy)*matrix->nnz,
                size_t(matrix->nnz)*sizeof(REAL),cudaMemcpyDeviceToHost,stream),"download matrix factors");
    }
    ~Impl() {
        if(stream) cudaStreamSynchronize(stream);
        if(graph_exec) cudaGraphExecDestroy(graph_exec);
        if(stream) cudaStreamDestroy(stream);
    }
};

GpuMatrix::GpuMatrix(const Symbolic_Matrix& a, unsigned slots, bool perturb, UpdateMode mode,SnapshotPolicy snapshot)
    : impl_(std::make_shared<Impl>(a, slots, perturb, mode,snapshot,std::vector<unsigned>{})) {}
GpuMatrix::GpuMatrix(const Symbolic_Matrix& a,const std::vector<unsigned>& positions,unsigned slots,bool perturb,UpdateMode mode,SnapshotPolicy snapshot)
    : impl_(std::make_shared<Impl>(a, slots, perturb, mode,snapshot,require_compact_order(positions))) {}
GpuMatrix::~GpuMatrix() = default;
unsigned GpuMatrix::tmp_slots() const { return impl_->slots; }
size_t GpuMatrix::task_device_bytes(unsigned copies) const {
    const unsigned pitch=impl_->value_pitch(copies);
    size_t bytes=checked_bytes(uint64_t(impl_->nnz)*copies,sizeof(REAL));
    const size_t scratch=checked_bytes(uint64_t(impl_->n)*impl_->slots,sizeof(REAL));
    bytes=checked_sum(bytes,checked_bytes(scratch,copies));
    if(pitch) bytes=checked_sum(bytes,checked_bytes(uint64_t(impl_->nnz)*pitch,sizeof(REAL)));
    bytes=checked_sum(bytes,checked_bytes(uint64_t(impl_->compact_elements)*copies,sizeof(REAL)));
    return bytes;
}
size_t GpuMatrix::task_host_bytes() const { return size_t(impl_->nnz)*sizeof(REAL); }
SnapshotPolicy GpuMatrix::snapshot_policy() const { return impl_->snapshot_policy; }
double GpuMatrix::snapshot_setup_ms() const { return impl_->snapshot_setup_ms; }
size_t GpuMatrix::compact_input_elements() const { return impl_->compact_elements; }
double GpuMatrix::compact_input_setup_ms() const { return impl_->compact_setup_ms; }
size_t GpuMatrix::shared_device_bytes() const { return impl_->shared_bytes; }
size_t GpuMatrix::update_map_bytes() const { return impl_->map_bytes; }
double GpuMatrix::update_map_setup_ms() const { return impl_->map_setup_ms; }
size_t GpuMatrix::update_plan_bytes() const { return impl_->plan_bytes; }
double GpuMatrix::update_plan_setup_ms() const { return impl_->plan_setup_ms; }
bool GpuMatrix::uses_native_mmac() const { return impl_->native_mmac && impl_->dense_tiles.bytes!=0; }
void GpuMatrix::select_device() {
    int count = 0;
    gpu_check(cudaGetDeviceCount(&count), "get device count");
    if (!count) throw std::runtime_error("No GPU detected");
    gpu_check(cudaSetDevice(0), "select device");
}
std::string GpuMatrix::device_name() const { return impl_->device; }
size_t GpuMatrix::free_device_bytes() {
    size_t free=0, total=0;
    gpu_check(cudaMemGetInfo(&free, &total), "query device memory");
    return free;
}
GpuTask::GpuTask(const GpuMatrix& matrix,unsigned batch) : impl_(new Impl(matrix.impl_,batch)) {}
unsigned GpuTask::batch_size() const { return impl_->copies; }
SnapshotPolicy GpuTask::snapshot_policy() const { return impl_->snapshot_policy; }
size_t GpuTask::snapshot_h2d_bytes() const {
    const auto& t=*impl_;
    if(t.snapshot_policy==SnapshotPolicy::Device)return 0;
    return t.snapshot_policy==SnapshotPolicy::Shared ? t.matrix->input.bytes : t.values.bytes;
}
size_t GpuTask::snapshot_h2d_ops() const {
    const auto& t=*impl_;
    if(t.snapshot_policy==SnapshotPolicy::Device)return 0;
    return t.snapshot_policy==SnapshotPolicy::PerCopy ? t.copies : 1;
}
size_t GpuTask::snapshot_restore_launches() const {
    const auto& t=*impl_;
    const bool restore=t.snapshot_policy==SnapshotPolicy::Device ||
        (t.snapshot_policy==SnapshotPolicy::Shared && t.copies>1);
    return size_t(restore)+size_t(t.soa_stride!=0);
}
GpuTask::~GpuTask() = default;

unsigned GpuTask::update_copy_tile_size() const { return impl_->soa_stride ? 1u<<impl_->copy_bits : 1; }
unsigned GpuTask::wide_copy_tile_size() const { return update_copy_tile_size(); }
unsigned GpuTask::soa_value_stride() const { return impl_->soa_stride; }
bool GpuTask::normalizes_each_level() const { return impl_->matrix->panel_mode || (impl_->soa_stride && normalize_soa_levels); }

size_t GpuTask::numeric_launches() const {
    const auto& t=*impl_;const auto& a=*t.matrix;
    if(a.update_mode==UpdateMode::Fused)return 1;
    // Input paths prepare the execution layout; numeric work converts only output.
    size_t count=t.soa_stride ? 1 : 0;
    if(a.panel_mode) {
        count+=a.levels.size()-1;
        for(const auto r:a.panel_immediate)count+=r.end>r.begin;
        for(const auto r:a.panel_deferred)count+=r.end>r.begin;
        for(const auto r:a.panel_dense)count+=r.end>r.begin;
        return count;
    }
    for(size_t lev=0;lev+1<a.levels.size();++lev) {
        const unsigned width=unsigned(a.levels[lev+1]-a.levels[lev]);
        if(a.update_mode==UpdateMode::Mapped) {
            const bool targets=a.target_prefix[a.levels[lev+1]]>a.target_prefix[a.levels[lev]];
            if(width>16)count+=t.soa_stride ? 1+targets : 1;
            else count+=targets+(t.soa_stride && normalize_soa_levels);
        } else if(width>16)count+=(width+a.slots-1)/a.slots;
        else for(int p=a.levels[lev];p<a.levels[lev+1];++p)
            count+=2+(a.csr_rows[a.columns[p]+1]>a.csr_diags[a.columns[p]]+1);
    }
    if(a.update_mode==UpdateMode::Mapped && a.deferred_columns_count && !(t.soa_stride && normalize_soa_levels))++count;
    return count;
}
size_t GpuTask::numeric_wide_blocks() const {
    const auto& t=*impl_;const auto& a=*t.matrix;if(a.panel_mode)return 0;
    uint64_t columns=0;
    for(size_t lev=0;lev+1<a.levels.size();++lev)if(a.levels[lev+1]-a.levels[lev]>16) {
        columns+=a.levels[lev+1]-a.levels[lev];
        if(t.soa_stride)columns+=a.target_prefix[a.levels[lev+1]]-a.target_prefix[a.levels[lev]];
    }
    return checked_bytes(columns,t.soa_stride ? t.copy_tiles(t.copy_bits) : t.copies);
}

void GpuTask::prepare_graph() {
    auto& t=*impl_;
    if(t.failed) throw std::logic_error("failed task must be rebuilt");
    if(t.pending || t.input_queued) throw std::logic_error("prepare graph before queuing work");
    if(t.graph_exec) return;
    gpu_check(cudaStreamBeginCapture(t.stream,cudaStreamCaptureModeThreadLocal),"begin graph capture");
    cudaGraph_t graph=nullptr;
    bool capturing=true;
    try {
        t.enqueue_work(true);
        const auto status=cudaStreamEndCapture(t.stream,&graph);
        capturing=false;
        gpu_check(status,"end graph capture");
        gpu_check(cudaGraphInstantiate(&t.graph_exec,graph,nullptr,nullptr,0),"instantiate task graph");
        cudaGraphDestroy(graph);
    } catch(...) {
        if(capturing) cudaStreamEndCapture(t.stream,&graph);
        if(graph) cudaGraphDestroy(graph);
        t.failed=true;
        throw;
    }
}
void GpuTask::upload_input() {
    auto& t=*impl_;
    if(t.failed || t.pending || t.input_queued) throw std::logic_error("cannot queue task input in this state");
    t.ready=false;
    try { t.enqueue_input();t.input_queued=true; }
    catch(...) { t.failed=true;cudaStreamSynchronize(t.stream);throw; }
}
void GpuTask::upload_input(const REAL* values,size_t elements) {
    auto& t=*impl_;
    if(!values || elements!=size_t(t.matrix->nnz)*t.copies)throw std::invalid_argument("input must contain factor_nnz * copies values");
    if(t.matrix->perturb)throw std::invalid_argument("fresh input requires an explicit perturbation threshold policy");
    if(t.failed || t.pending || t.input_queued)throw std::logic_error("cannot queue task input in this state");
    t.ready=false;
    try {t.enqueue_input(values);t.input_queued=true;}
    // An upload may have queued work. Drain before the caller can release
    // the external input; a cleanup error must not replace the original error.
    catch(...) {t.failed=true;cudaStreamSynchronize(t.stream);throw;}
}
void GpuTask::upload_compact_input(const REAL* values,size_t elements) {
    auto& t=*impl_;
    if(!t.matrix->compact_elements || !values || elements!=size_t(t.matrix->compact_elements)*t.copies)
        throw std::invalid_argument("compact input must match a prepared compact order and batch size");
    if(t.matrix->perturb)throw std::invalid_argument("fresh input requires an explicit perturbation threshold policy");
    if(t.failed || t.pending || t.input_queued)throw std::logic_error("cannot queue task input in this state");
    t.ready=false;
    try {t.enqueue_compact(values);t.input_queued=true;}
    catch(...) {t.failed=true;cudaStreamSynchronize(t.stream);throw;}
}
void GpuTask::enqueue() {
    auto& t=*impl_;
    if(t.failed) throw std::logic_error("failed task must be rebuilt");
    if(t.pending) throw std::logic_error("wait for the task before reusing it");
    t.pending=true;t.ready=false;
    try {
        if(!t.input_queued) t.enqueue_input();
        t.input_queued=false;
        if(t.graph_exec) gpu_check(cudaGraphLaunch(t.graph_exec,t.stream),"launch task graph");
        else t.enqueue_work(false);
    } catch(...) { t.failed=true;cudaStreamSynchronize(t.stream);throw; }
}
void GpuTask::wait() {
    auto& t = *impl_;
    if(t.input_queued) throw std::logic_error("enqueue numeric work before waiting for a result");
    if (t.pending) {
        const auto status = cudaStreamSynchronize(t.stream);
        t.pending = false;
        if (status != cudaSuccess) {
            t.failed = true;
            gpu_check(status, "wait for task");
        }
        if (!t.failed) t.ready = true;
    }
    if (t.failed) throw std::runtime_error("task submission failed; rebuild task before reuse");
}
const REAL* GpuTask::factors(unsigned copy) const {
    if (copy >= impl_->copies) throw std::out_of_range("task copy index");
    if (impl_->failed || !impl_->ready) throw std::logic_error("task results are not ready");
    return impl_->output.ptr + size_t(copy)*impl_->matrix->nnz;
}
void GpuTask::check() const {
    const auto& a = *impl_->matrix;
    for (unsigned copy=0; copy<impl_->copies; ++copy) {
        const REAL* values = factors(copy);
        for (unsigned p=0; p<a.nnz; ++p)
            if (!std::isfinite(values[p])) throw std::runtime_error("LU data check: NaN/Inf found");
        for (unsigned j=0; j<a.n; ++j)
            if (values[a.diagonals[j]] == REAL(0)) throw std::runtime_error("LU data check: zero diagonal");
    }
}

struct GpuTaskSet::Impl {
    std::vector<std::shared_ptr<const GpuMatrix::Impl>> templates;
    std::vector<size_t> task_templates,full_offsets{0},compact_offsets{0};
    Buffer<REAL> values,compact_values;
    Buffer<REAL,true> output,default_input;
    Buffer<FusedTaskDescriptor> tasks;
    unsigned block_threads=256;
    cudaStream_t stream=nullptr;
    bool all_compact=true,perturb=false,compact_queued=false;
    bool pending=false,ready=false,failed=false,input_queued=false;
    double default_setup_ms=0;

    Impl(const std::vector<const GpuMatrix*>& matrices,const std::vector<size_t>& indices)
        : task_templates(indices) {
        if(indices.empty())throw std::invalid_argument("task set must not be empty");
        for(const auto* matrix:matrices) {
            if(!matrix || matrix->impl_->update_mode!=UpdateMode::Fused)
                throw std::invalid_argument("task set requires Fused templates");
            templates.push_back(matrix->impl_);
        }
        std::vector<FusedTaskDescriptor> descriptors;
        for(size_t index:indices) {
            if(index>=templates.size())throw std::out_of_range("task template index");
            const auto& a=*templates[index];
            block_threads=std::max(block_threads,a.fused_block_threads);
            if(indices.size()>a.max_grid_x)throw std::invalid_argument("task set exceeds device grid limit");
            descriptors.push_back({full_offsets.back(),compact_offsets.back(),a.nnz,a.compact_elements,
                unsigned(a.levels.size()-1),a.diag.ptr,a.input_positions.ptr,a.level_idx.ptr,
                a.fused_steps.ptr,a.panel_groups.ptr,a.panel_records.ptr,a.panel_pairs.ptr,a.fused_strips.ptr,a.fused_destinations.ptr,a.fused_normalizations.ptr,a.inline_singletons,
                a.perturb,a.threshold});
            full_offsets.push_back(checked_sum(full_offsets.back(),a.nnz));
            compact_offsets.push_back(checked_sum(compact_offsets.back(),a.compact_elements));
            all_compact=all_compact && a.compact_elements!=0;
            perturb=perturb || a.perturb;
        }
        values.allocate(full_offsets.back());output.allocate(full_offsets.back());
        if(all_compact)compact_values.allocate(compact_offsets.back());
        std::stable_sort(descriptors.begin(),descriptors.end(),[](const auto& x,const auto& y){return x.nnz>y.nnz;});
        tasks.upload(descriptors);
        gpu_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create task set stream");
    }
    ~Impl() {
        if(stream) {cudaStreamSynchronize(stream);cudaStreamDestroy(stream);}
    }
    void require_upload() const {
        if(failed || pending || input_queued)throw std::logic_error("cannot queue task set input in this state");
    }
    void upload(const REAL* input,bool compact) {
        require_upload();ready=false;
        try {
            auto& destination=compact ? compact_values : values;
            gpu_check(cudaMemcpyAsync(destination.ptr,input,destination.bytes,cudaMemcpyHostToDevice,stream),"upload task set input");
            compact_queued=compact;input_queued=true;
        } catch(...) {failed=true;cudaStreamSynchronize(stream);throw;}
    }
    void upload_default() {
        require_upload();
        if(!default_input.ptr) {
            const auto start=std::chrono::steady_clock::now();
            default_input.allocate(full_offsets.back());
            for(size_t task=0;task<task_templates.size();++task) {
                const auto& a=*templates[task_templates[task]];
                std::copy(a.input.ptr,a.input.ptr+a.nnz,default_input.ptr+full_offsets[task]);
            }
            default_setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        }
        upload(default_input.ptr,false);
    }
};

GpuTaskSet::GpuTaskSet(const std::vector<const GpuMatrix*>& matrices,const std::vector<size_t>& indices)
    : impl_(new Impl(matrices,indices)) {}
GpuTaskSet::~GpuTaskSet()=default;
size_t GpuTaskSet::task_count() const {return impl_->task_templates.size();}
size_t GpuTaskSet::factor_offset(size_t task) const {return impl_->full_offsets.at(task);}
size_t GpuTaskSet::compact_input_offset(size_t task) const {return impl_->compact_offsets.at(task);}
size_t GpuTaskSet::device_bytes() const {return checked_sum(checked_sum(impl_->values.bytes,impl_->compact_values.bytes),impl_->tasks.bytes);}
size_t GpuTaskSet::pinned_bytes() const {return checked_sum(impl_->output.bytes,impl_->default_input.bytes);}
double GpuTaskSet::default_input_setup_ms() const {return impl_->default_setup_ms;}
void GpuTaskSet::upload_input() {impl_->upload_default();}
void GpuTaskSet::upload_input(const REAL* input,size_t elements) {
    if(!input || elements!=impl_->full_offsets.back())throw std::invalid_argument("input must match task set factor slots");
    if(impl_->perturb)throw std::invalid_argument("fresh input requires an explicit perturbation threshold policy");
    impl_->upload(input,false);
}
void GpuTaskSet::upload_compact_input(const REAL* input,size_t elements) {
    if(!impl_->all_compact || !input || elements!=impl_->compact_offsets.back())
        throw std::invalid_argument("compact input must match all prepared task set input positions");
    if(impl_->perturb)throw std::invalid_argument("fresh input requires an explicit perturbation threshold policy");
    impl_->upload(input,true);
}
void GpuTaskSet::enqueue() {
    auto& t=*impl_;
    if(t.failed || t.pending)throw std::logic_error("task set is failed or pending");
    if(!t.input_queued)t.upload_default();
    t.ready=false;t.pending=true;t.input_queued=false;
    try {
        fused_task_set<<<unsigned(task_count()),t.block_threads,0,t.stream>>>(t.values.ptr,t.compact_values.ptr,t.tasks.ptr,t.compact_queued);
        gpu_check(cudaGetLastError(),"factor task set");
        gpu_check(cudaMemcpyAsync(t.output.ptr,t.values.ptr,t.output.bytes,cudaMemcpyDeviceToHost,t.stream),"download task set factors");
    } catch(...) {t.failed=true;cudaStreamSynchronize(t.stream);throw;}
}
void GpuTaskSet::wait() {
    auto& t=*impl_;
    if(t.input_queued)throw std::logic_error("enqueue task set numeric work before waiting");
    if(t.pending) {
        const auto status=cudaStreamSynchronize(t.stream);t.pending=false;
        if(status!=cudaSuccess) {t.failed=true;gpu_check(status,"wait for task set");}
        if(!t.failed)t.ready=true;
    }
    if(t.failed)throw std::runtime_error("task set submission failed; rebuild before reuse");
}
const REAL* GpuTaskSet::factors(size_t task) const {
    if(task>=task_count())throw std::out_of_range("task set result index");
    if(impl_->failed || !impl_->ready)throw std::logic_error("task set results are not ready");
    return impl_->output.ptr+impl_->full_offsets[task];
}
void GpuTaskSet::check() const {
    for(size_t task=0;task<task_count();++task) {
        const auto& a=*impl_->templates[impl_->task_templates[task]];
        const REAL* data=factors(task);
        for(unsigned p=0;p<a.nnz;++p)if(!std::isfinite(data[p]))throw std::runtime_error("LU data check: NaN/Inf found");
        for(unsigned p:a.diagonals)if(data[p]==REAL(0))throw std::runtime_error("LU data check: zero diagonal");
    }
}

bool LUonDevice(Symbolic_Matrix& a, ostream& out, ostream& err, bool perturb)
{
    try {
        const auto start = std::chrono::steady_clock::now();
        GpuMatrix matrix(a, 32, perturb);
        GpuTask task(matrix);
        task.enqueue(); task.wait(); task.check();
        std::copy(task.factors(), task.factors()+a.nnz, a.val.begin());
        const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        out << "Device 0: " << matrix.device_name() << "\nPrepare and factor: " << ms << " ms\n";
        return true;
    } catch (const std::exception& error) {
        err << error.what() << '\n';
        return false;
    }
}
