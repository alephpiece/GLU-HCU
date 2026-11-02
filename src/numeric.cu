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

using namespace std;


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
        const int inLevPos)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / GLU_GROUP_SIZE;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[size_t(bid)*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __syncthreads();

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

__global__ void RL_perturb(
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
        const REAL pert)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = threadIdx.x / GLU_GROUP_SIZE;

    const unsigned currentCol = level_idx_dev[levelHead+inLevPos+bid];
    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    if (tid == 0 && fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;
    __syncthreads();

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[size_t(bid)*n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
    __syncthreads();

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

__global__ void RL_onecol_factorizeCurrentCol(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[size_t(stream) * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_factorizeCurrentCol_perturb(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n,
        const REAL pert)
{
    const int tid = threadIdx.x;

    const unsigned currentLColSize = sym_c_ptr_dev[currentCol + 1] - l_col_ptr_dev[currentCol] - 1;
    const unsigned currentLPos = l_col_ptr_dev[currentCol] + tid + 1;

    if (tid == 0 && fabs(val_dev[l_col_ptr_dev[currentCol]]) < pert)
        val_dev[l_col_ptr_dev[currentCol]] = pert;
    __syncthreads();

    //update current col

    int offset = 0;
    while (currentLColSize > offset)
    {
        if (tid + offset < currentLColSize)
        {
            unsigned ridx = sym_r_idx_dev[currentLPos + offset];

            val_dev[currentLPos + offset] /= val_dev[l_col_ptr_dev[currentCol]];
            tmpMem[size_t(stream) * n + ridx]= val_dev[currentLPos + offset];
        }
        offset += blockDim.x;
    }
}

__global__ void RL_onecol_updateSubmat(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        REAL* __restrict__ val_dev,
        const unsigned* __restrict__ csr_c_idx_dev,
        const unsigned* __restrict__ csr_diag_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    __shared__ REAL s;

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

__global__ void RL_onecol_cleartmpMem(
        const unsigned* __restrict__ sym_c_ptr_dev,
        const unsigned* __restrict__ sym_r_idx_dev,
        const unsigned* __restrict__ l_col_ptr_dev,
        const unsigned currentCol,
        REAL* __restrict__ tmpMem,
        const int stream,
        const unsigned n)
{
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
void gpu_check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<class T, bool Pinned = false> struct Buffer {
    T* ptr = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (ptr) { if (Pinned) cudaFreeHost(ptr); else cudaFree(ptr); } }
    void allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::overflow_error("GPU allocation size overflow");
        if (Pinned) gpu_check(cudaMallocHost(reinterpret_cast<void**>(&ptr), count*sizeof(T)), "allocate pinned buffer");
        else gpu_check(cudaMalloc(reinterpret_cast<void**>(&ptr), count*sizeof(T)), "allocate device buffer");
    }
    void upload(const std::vector<T>& input) {
        allocate(input.size());
        gpu_check(cudaMemcpy(ptr, input.data(), input.size()*sizeof(T), cudaMemcpyHostToDevice), "upload structure");
    }
};
}

struct GpuMatrix::Impl {
    unsigned n, nnz, slots;
    bool perturb;
    REAL threshold = 0;
    std::string device;
    std::vector<int> levels, columns;
    std::vector<unsigned> csr_rows, csr_diags, diagonals;
    Buffer<unsigned> col, row, diag, csr_row, csr_col, csr_diag;
    Buffer<int> level_idx;
    Buffer<REAL, true> input;

    Impl(const Symbolic_Matrix& a, unsigned requested_slots, bool use_perturb)
        : n(a.n), nnz(a.nnz), slots(0), perturb(use_perturb),
          levels(a.level_ptr), columns(a.level_idx), csr_rows(a.csr_r_ptr),
          csr_diags(a.csr_diag_ptr), diagonals(a.l_col_ptr)
    {
        if (!n || !nnz || !requested_slots || levels.size() < 2)
            throw std::invalid_argument("empty matrix or zero scratch slots");
        int count = 0;
        gpu_check(cudaGetDeviceCount(&count), "get device count");
        if (!count) throw std::runtime_error("No GPU detected");
        gpu_check(cudaSetDevice(0), "select device");
        cudaDeviceProp prop;
        gpu_check(cudaGetDeviceProperties(&prop, 0), "get device properties");
        if ((prop.warpSize != 32 && prop.warpSize != 64) || prop.maxThreadsPerBlock < 1024)
            throw std::runtime_error("requires warp32/wave64 and 1024-thread blocks");
        device = prop.name;
        unsigned needed = 1;
        for (size_t i=1; i<levels.size(); ++i) {
            const unsigned width = unsigned(levels[i]-levels[i-1]);
            // Small levels execute serially in slot zero.
            if (width > 16) needed = std::max(needed, width);
        }
        slots = std::min(requested_slots, needed);
        col.upload(a.sym_c_ptr); row.upload(a.sym_r_idx); diag.upload(a.l_col_ptr);
        csr_row.upload(a.csr_r_ptr); csr_col.upload(a.csr_c_idx); csr_diag.upload(a.csr_diag_ptr);
        level_idx.upload(a.level_idx);
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
};

struct GpuTask::Impl {
    std::shared_ptr<const GpuMatrix::Impl> matrix;
    Buffer<REAL> values, scratch;
    Buffer<REAL, true> output;
    cudaStream_t stream = nullptr;
    bool pending = false, ready = false, failed = false;
    explicit Impl(std::shared_ptr<const GpuMatrix::Impl> m) : matrix(std::move(m)) {
        values.allocate(matrix->nnz);
        scratch.allocate(size_t(matrix->n)*matrix->slots);
        output.allocate(matrix->nnz);
        gpu_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create task stream");
        try {
            gpu_check(cudaMemsetAsync(scratch.ptr, 0, size_t(matrix->n)*matrix->slots*sizeof(REAL), stream),
                "initialize scratch");
            gpu_check(cudaStreamSynchronize(stream), "finish task preparation");
        } catch (...) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            throw;
        }
    }
    ~Impl() {
        // Also drain partially submitted work after an exception, before any
        // buffers (including the retained template) can be released.
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
};

GpuMatrix::GpuMatrix(const Symbolic_Matrix& a, unsigned slots, bool perturb)
    : impl_(std::make_shared<Impl>(a, slots, perturb)) {}
GpuMatrix::~GpuMatrix() = default;
unsigned GpuMatrix::tmp_slots() const { return impl_->slots; }
size_t GpuMatrix::task_device_bytes() const {
    const size_t scratch = size_t(impl_->n)*impl_->slots;
    if (scratch > std::numeric_limits<size_t>::max()/sizeof(REAL)-impl_->nnz)
        throw std::overflow_error("task memory size overflow");
    return (size_t(impl_->nnz) + scratch)*sizeof(REAL);
}
size_t GpuMatrix::task_host_bytes() const { return size_t(impl_->nnz)*sizeof(REAL); }
std::string GpuMatrix::device_name() const { return impl_->device; }
size_t GpuMatrix::free_device_bytes() {
    size_t free=0, total=0;
    gpu_check(cudaMemGetInfo(&free, &total), "query device memory");
    return free;
}
GpuTask::GpuTask(const GpuMatrix& matrix) : impl_(new Impl(matrix.impl_)) {}
GpuTask::~GpuTask() = default;

void GpuTask::enqueue()
{
    auto& t = *impl_;
    const auto& a = *t.matrix;
    if (t.failed) throw std::logic_error("failed task must be rebuilt");
    if (t.pending) throw std::logic_error("wait for the task before reusing it");
    t.pending = true;
    t.ready = false;
    try {
        gpu_check(cudaMemcpyAsync(t.values.ptr, a.input.ptr, size_t(a.nnz)*sizeof(REAL),
            cudaMemcpyHostToDevice, t.stream), "restore matrix values");
        for (size_t lev=0; lev+1<a.levels.size(); ++lev) {
            const int width = a.levels[lev+1]-a.levels[lev];
            if (width > 16) {
                const unsigned threads = width > 896 ? 64 : width > 448 ? 128 : 1024;
                for (int offset=0; offset<width;) {
                    const unsigned blocks = std::min(unsigned(width-offset), a.slots);
                    if (!a.perturb)
                        RL<<<blocks,threads,0,t.stream>>>(a.col.ptr,a.row.ptr,t.values.ptr,a.diag.ptr,
                            a.csr_row.ptr,a.csr_col.ptr,a.csr_diag.ptr,a.level_idx.ptr,t.scratch.ptr,
                            a.n,a.levels[lev],offset);
                    else
                        RL_perturb<<<blocks,threads,0,t.stream>>>(a.col.ptr,a.row.ptr,t.values.ptr,a.diag.ptr,
                            a.csr_row.ptr,a.csr_col.ptr,a.csr_diag.ptr,a.level_idx.ptr,t.scratch.ptr,
                            a.n,a.levels[lev],offset,a.threshold);
                    gpu_check(cudaGetLastError(), "launch batched level");
                    offset += int(blocks);
                }
            } else {
                for (int p=a.levels[lev]; p<a.levels[lev+1]; ++p) {
                    const unsigned column = a.columns[p];
                    const unsigned targets = a.csr_rows[column+1]-a.csr_diags[column]-1;
                    if (!a.perturb)
                        RL_onecol_factorizeCurrentCol<<<1,1024,0,t.stream>>>(a.col.ptr,a.row.ptr,
                            t.values.ptr,a.diag.ptr,column,t.scratch.ptr,0,a.n);
                    else
                        RL_onecol_factorizeCurrentCol_perturb<<<1,1024,0,t.stream>>>(a.col.ptr,a.row.ptr,
                            t.values.ptr,a.diag.ptr,column,t.scratch.ptr,0,a.n,a.threshold);
                    gpu_check(cudaGetLastError(), "launch column factorization");
                    if (targets) {
                        RL_onecol_updateSubmat<<<targets,1024,0,t.stream>>>(a.col.ptr,a.row.ptr,t.values.ptr,
                            a.csr_col.ptr,a.csr_diag.ptr,column,t.scratch.ptr,0,a.n);
                        gpu_check(cudaGetLastError(), "launch column update");
                    }
                    RL_onecol_cleartmpMem<<<1,1024,0,t.stream>>>(a.col.ptr,a.row.ptr,a.diag.ptr,
                        column,t.scratch.ptr,0,a.n);
                    gpu_check(cudaGetLastError(), "launch scratch clear");
                }
            }
        }
        gpu_check(cudaMemcpyAsync(t.output.ptr,t.values.ptr,size_t(a.nnz)*sizeof(REAL),
            cudaMemcpyDeviceToHost,t.stream), "download factors");
    } catch (...) {
        t.failed = true;
        throw;
    }
}

void GpuTask::wait() {
    auto& t = *impl_;
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
const REAL* GpuTask::factors() const {
    if (impl_->failed || !impl_->ready) throw std::logic_error("task results are not ready");
    return impl_->output.ptr;
}
void GpuTask::check() const {
    const REAL* values = factors();
    const auto& a = *impl_->matrix;
    for (unsigned p=0; p<a.nnz; ++p)
        if (!std::isfinite(values[p])) throw std::runtime_error("LU data check: NaN/Inf found");
    for (unsigned j=0; j<a.n; ++j)
        if (values[a.diagonals[j]] == REAL(0)) throw std::runtime_error("LU data check: zero diagonal");
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
