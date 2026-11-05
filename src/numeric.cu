#include "gpu-compat.h"
#include "numeric.h"
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

// A wide level keeps source columns independent. Within a source, adjacent
// lanes own copies; the remaining slots divide rows and independent targets.
template<bool Perturb = false>
__global__ void mapped_wide_soa(
        REAL* values,unsigned copies,const unsigned* col,const unsigned* diag,
        const unsigned* csr_row,const unsigned* csr_diag,const int* order,int begin,
        const uint64_t* bases,const unsigned* destinations,const unsigned* pivots,REAL threshold,unsigned copy_bits=5,
        unsigned value_stride=0)
{
    if(!value_stride) value_stride=copies;
    __shared__ REAL divisors[32];
    const unsigned source=unsigned(order[begin+blockIdx.x]);
    const unsigned diagonal=diag[source],lower=diagonal+1,count=col[source+1]-lower;
    const unsigned copy_lanes=1u<<copy_bits,row_slots=blockDim.x>>copy_bits;
    const unsigned copy_lane=threadIdx.x&(copy_lanes-1),slot=threadIdx.x>>copy_bits;
    const unsigned copy=blockIdx.y*copy_lanes+copy_lane;
    const bool valid=copy<copies;
    if(slot==0 && valid) {
        REAL divisor=values[size_t(diagonal)*value_stride+copy];
        if constexpr(Perturb) {
            if(fabs(divisor)<threshold) {
                divisor=threshold;
                values[size_t(diagonal)*value_stride+copy]=divisor;
            }
        }
        divisors[copy_lane]=divisor;
    }
    __syncthreads();
    unsigned row_bits=0;
    while((1u<<row_bits)<count && (1u<<row_bits)<row_slots) ++row_bits;
    const unsigned row_lanes=1u<<row_bits;
    if(valid && slot<row_lanes)
        for(unsigned i=slot;i<count;i+=row_lanes)
            values[size_t(lower+i)*value_stride+copy]/=divisors[copy_lane];
    // Invalid copy lanes also participate. Every consumer sees the normalized
    // lower values, including consumers in a different target group.
    __syncthreads();
    if(!valid || !count) return;
    const unsigned row_lane=slot&(row_lanes-1),target_lane=slot>>row_bits;
    const unsigned target_groups=row_slots>>row_bits,first=csr_diag[source]+1;
    for(unsigned q=first+target_lane;q<csr_row[source+1];q+=target_groups) {
        const REAL pivot=values[size_t(pivots[q])*value_stride+copy];
        const uint64_t base=bases[source]+uint64_t(q-first)*count;
        for(unsigned i=row_lane;i<count;i+=row_lanes) {
            const REAL factor=values[size_t(lower+i)*value_stride+copy];
            if(factor!=REAL(0))
                atomicAdd(&values[size_t(destinations[base+i])*value_stride+copy],-factor*pivot);
        }
    }
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
// Copy lanes fill naturally up to the fixed kernel cap. This is also the
// physical row alignment, so tiny batches do not pay a 32-column allocation.
unsigned filled_copy_bits(unsigned copies,unsigned cap) {
    unsigned bits=0;
    while(bits<cap && (1u<<bits)<copies) ++bits;
    return bits;
}
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

struct GpuMatrix::Impl {
    unsigned n, nnz, slots, max_batch, max_grid_x;
    unsigned deferred_columns_count=0;
    bool perturb;
    UpdateMode update_mode;
    size_t shared_bytes = 0, map_bytes = 0;
    double map_setup_ms = 0;
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
    Buffer<REAL, true> input;

    unsigned value_pitch(unsigned copies) const {
        if(!copies || copies>max_batch) throw std::invalid_argument("batch size exceeds device grid limit");
        if(update_mode!=UpdateMode::Mapped || copies==1) return 0;
        const unsigned lanes=1u<<filled_copy_bits(copies,5);
        const uint64_t pitch=(uint64_t(copies)+lanes-1)/lanes*lanes;
        if(pitch>std::numeric_limits<unsigned>::max()) throw std::overflow_error("SoA leading dimension overflow");
        return unsigned(pitch);
    }

    Impl(const Symbolic_Matrix& a, unsigned requested_slots, bool use_perturb, UpdateMode mode)
        : n(a.n), nnz(a.nnz), slots(0), perturb(use_perturb), update_mode(mode),
          levels(a.level_ptr), columns(a.level_idx), csr_rows(a.csr_r_ptr),
          csr_diags(a.csr_diag_ptr), diagonals(a.l_col_ptr)
    {
        if (!n || !nnz || !requested_slots || levels.size() < 2)
            throw std::invalid_argument("empty matrix or zero scratch slots");
        if (mode != UpdateMode::Scan && mode != UpdateMode::Mapped)
            throw std::invalid_argument("invalid update mode");
        GpuMatrix::select_device();
        cudaDeviceProp prop;
        gpu_check(cudaGetDeviceProperties(&prop, 0), "get device properties");
        if ((prop.warpSize != 32 && prop.warpSize != 64) || prop.maxThreadsPerBlock < 1024)
            throw std::runtime_error("requires warp32/wave64 and 1024-thread blocks");
        device = prop.name;
        max_batch = unsigned(prop.maxGridSize[1]);
        max_grid_x = unsigned(prop.maxGridSize[0]);
        unsigned needed = 1;
        for (size_t i=1; i<levels.size(); ++i) {
            const unsigned width = unsigned(levels[i]-levels[i-1]);
            if (mode == UpdateMode::Mapped && width > 16 && width > unsigned(prop.maxGridSize[0]))
                throw std::invalid_argument("mapped level width exceeds device grid limit");
            // Small levels execute serially in slot zero.
            if (mode == UpdateMode::Scan && width > 16) needed = std::max(needed, width);
        }
        slots = mode == UpdateMode::Scan ? std::min(requested_slots, needed) : 0;
        col.upload(a.sym_c_ptr); diag.upload(a.l_col_ptr);
        csr_row.upload(a.csr_r_ptr); csr_diag.upload(a.csr_diag_ptr);
        if(mode==UpdateMode::Scan) { row.upload(a.sym_r_idx); csr_col.upload(a.csr_c_idx); }
        level_idx.upload(a.level_idx);
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
                             csr_col.bytes, csr_diag.bytes, level_idx.bytes, map_bytes, level_targets.bytes, deferred_columns.bytes})
            shared_bytes = checked_sum(shared_bytes, bytes);
        input.allocate(nnz);
        std::copy(a.val.begin(), a.val.end(), input.ptr);
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
    Buffer<REAL> values, scratch, interleaved_values;
    Buffer<REAL, true> output;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    unsigned copies,copy_bits,wide_copy_bits,soa_stride;
    bool pending=false,ready=false,failed=false,input_queued=false;
    unsigned copy_tiles(unsigned bits) const { return (copies+(1u<<bits)-1)>>bits; }
    explicit Impl(std::shared_ptr<const GpuMatrix::Impl> m,unsigned batch)
        : matrix(std::move(m)),copies(batch),copy_bits(filled_copy_bits(batch,5)),
          wide_copy_bits(filled_copy_bits(batch,3)),soa_stride(matrix->value_pitch(batch)) {
        values.allocate(matrix->nnz,copies);
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
    void enqueue_input() {
        for(unsigned copy=0;copy<copies;++copy)
            gpu_check(cudaMemcpyAsync(values.ptr+size_t(copy)*matrix->nnz,matrix->input.ptr,
                size_t(matrix->nnz)*sizeof(REAL),cudaMemcpyHostToDevice,stream),"upload matrix input");
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
        if(soa) transpose(true);
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
                            mapped_level_update<Perturb><<<dim3(unsigned(targets),copies),column_threads,0,stream>>>(
                                values.ptr,a.nnz,a.col.ptr,a.diag.ptr,a.csr_diag.ptr,a.level_idx.ptr,a.level_targets.ptr,
                                begin,end,a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr,a.threshold);
                        gpu_check(cudaGetLastError(),"update mapped level");
                    }
                    continue;
                }
                if(soa) {
                    mapped_wide_soa<Perturb><<<dim3(width,copy_tiles(wide_copy_bits)),256,0,stream>>>(
                        interleaved_values.ptr,copies,a.col.ptr,a.diag.ptr,a.csr_row.ptr,a.csr_diag.ptr,a.level_idx.ptr,begin,
                        a.map_bases.ptr,a.map_destinations.ptr,a.map_pivots.ptr,a.threshold,wide_copy_bits,soa_stride);
                    gpu_check(cudaGetLastError(),"update wide mapped level");
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
        if(matrix->update_mode==UpdateMode::Mapped) {
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

GpuMatrix::GpuMatrix(const Symbolic_Matrix& a, unsigned slots, bool perturb, UpdateMode mode)
    : impl_(std::make_shared<Impl>(a, slots, perturb, mode)) {}
GpuMatrix::~GpuMatrix() = default;
unsigned GpuMatrix::tmp_slots() const { return impl_->slots; }
size_t GpuMatrix::task_device_bytes(unsigned copies) const {
    const unsigned pitch=impl_->value_pitch(copies);
    size_t bytes=checked_bytes(uint64_t(impl_->nnz)*copies,sizeof(REAL));
    const size_t scratch=checked_bytes(uint64_t(impl_->n)*impl_->slots,sizeof(REAL));
    bytes=checked_sum(bytes,checked_bytes(scratch,copies));
    if(pitch) bytes=checked_sum(bytes,checked_bytes(uint64_t(impl_->nnz)*pitch,sizeof(REAL)));
    return bytes;
}
size_t GpuMatrix::task_host_bytes() const { return size_t(impl_->nnz)*sizeof(REAL); }
size_t GpuMatrix::shared_device_bytes() const { return impl_->shared_bytes; }
size_t GpuMatrix::update_map_bytes() const { return impl_->map_bytes; }
double GpuMatrix::update_map_setup_ms() const { return impl_->map_setup_ms; }
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
GpuTask::~GpuTask() = default;

unsigned GpuTask::update_copy_tile_size() const { return impl_->soa_stride ? 1u<<impl_->copy_bits : 1; }
unsigned GpuTask::wide_copy_tile_size() const { return impl_->soa_stride ? 1u<<impl_->wide_copy_bits : 1; }
unsigned GpuTask::soa_value_stride() const { return impl_->soa_stride; }
bool GpuTask::normalizes_each_level() const { return impl_->soa_stride && normalize_soa_levels; }

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
    catch(...) { t.failed=true;throw; }
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
    } catch(...) { t.failed=true;throw; }
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
