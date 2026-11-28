#include "scale-certificate.h"

namespace diagonal_scale
{

constexpr unsigned threads = 256;
constexpr unsigned default_tiles = 16;

// Zero disables reuse. Both sizes are known when the template is constructed.
inline unsigned minimum_fill_ratio()
{
    const char* value = std::getenv("GLU_READY_DIAGONAL_SCALE");
    if (value && std::strcmp(value, "1") == 0)
        return 1;
    if (value && std::strcmp(value, "fill2") == 0)
        return 2;
    return 0;
}

inline unsigned selected_tiles(unsigned limit)
{
    const char* value = std::getenv("GLU_READY_SCALE_TILES");
    if (!value)
        return std::min(default_tiles, limit);
    char* end = nullptr;
    const auto count = std::strtoul(value, &end, 10);
    if (end == value || *end || !count || count > limit)
        throw std::invalid_argument("GLU_READY_SCALE_TILES exceeds device grid limits");
    return unsigned(count);
}

struct Template
{
    Buffer<RowAnchors> anchors;
    Buffer<uint32_t> factor_coordinates;
    Buffer<REAL> base, factors;

    size_t bytes() const { return anchors.bytes + factor_coordinates.bytes + base.bytes + factors.bytes; }
};

struct Lookup
{
    Buffer<uint32_t> heads;
    Buffer<uint16_t> pairs;
    Buffer<REAL> reciprocals;

    Lookup()
    {
        const ProductTable table;
        heads.upload(table.heads);
        pairs.upload(table.pairs);
        reciprocals.upload(table.reciprocals);
    }

    size_t bytes() const { return heads.bytes + pairs.bytes + reciprocals.bytes; }
};

// All templates use logical device 0. Keep one table while any set uses it,
// and release its device pointers before an idle application's device reset.
inline std::shared_ptr<const Lookup> lookup()
{
    static std::mutex mutex;
    static std::weak_ptr<const Lookup> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto table = cache.lock();
    if (!table)
    {
        table = std::make_shared<const Lookup>();
        cache = table;
    }
    return table;
}

struct Job
{
    unsigned n = 0;
    size_t code_offset = 0;
    const RowAnchors* anchors = nullptr;
    const uint32_t* factor_coordinates = nullptr;
    const REAL *base = nullptr, *base_factors = nullptr;
};

template <class Real> __device__ bool equal_bits(Real a, Real b)
{
    if constexpr (sizeof(Real) == sizeof(float))
        return __float_as_uint(a) == __float_as_uint(b);
    else
        return __double_as_longlong(a) == __double_as_longlong(b);
}

__global__ void recover_rows(const REAL* values,
    const FusedTaskDescriptor* tasks,
    const Job* jobs,
    uint16_t* codes,
    unsigned* rejected,
    const uint32_t* heads,
    const uint16_t* pairs)
{
    const auto job = jobs[blockIdx.x];
    if (!job.n)
    {
        if (threadIdx.x == 0 && blockIdx.y == 0)
            rejected[blockIdx.x] = 1;
        return;
    }
    const auto* input = values + tasks[blockIdx.x].compact_offset;
    bool failed = false;
    for (unsigned row = blockDim.x * blockIdx.y + threadIdx.x; row < job.n; row += blockDim.x * gridDim.y)
    {
        uint16_t pair = 0;
        const bool found = recover_row(job.anchors[row], heads, pairs, input, job.base, pair);
        codes[job.code_offset + row] = pair;
        failed |= !found;
    }
    if (__syncthreads_or(failed) && threadIdx.x == 0)
        atomicExch(rejected + blockIdx.x, 1u);
}

__global__ void verify_inputs(const REAL* values,
    const FusedTaskDescriptor* tasks,
    const Job* jobs,
    const uint16_t* codes,
    unsigned* rejected)
{
    const auto job = jobs[blockIdx.x];
    if (!job.n)
        return;
    const auto task = tasks[blockIdx.x];
    const auto* input = values + task.compact_offset;
    const auto* scales = codes + job.code_offset;
    bool mismatch = false;
    for (unsigned p = blockDim.x * blockIdx.y + threadIdx.x; p < task.compact_count; p += blockDim.x * gridDim.y)
    {
        const uint32_t coordinate = job.factor_coordinates[task.positions[p]];
        const REAL expected = rebuild_input(job.base[p], coordinate, scales);
        mismatch |= !equal_bits(expected, input[p]);
    }
    if (__syncthreads_or(mismatch) && threadIdx.x == 0)
        atomicExch(rejected + blockIdx.x, 1u);
}

__global__ void write_factors(REAL* values,
    const FusedTaskDescriptor* tasks,
    const Job* jobs,
    const uint16_t* codes,
    const unsigned* rejected,
    const REAL* reciprocals)
{
    if (rejected[blockIdx.x])
        return;
    const auto job = jobs[blockIdx.x];
    const auto task = tasks[blockIdx.x];
    auto* factors = values + task.factor_offset;
    const auto* scales = codes + job.code_offset;
    for (unsigned p = blockDim.x * blockIdx.y + threadIdx.x; p < task.nnz; p += blockDim.x * gridDim.y)
        factors[p] = rebuild_factor(job.base_factors[p], job.factor_coordinates[p], scales, reciprocals);
}

// Keep the normal input scatter, update plan and output contract for any
// matrix whose entire compact input was not certified.
__global__ void fallback(REAL* factors, const REAL* input, const FusedTaskDescriptor* tasks, const unsigned* rejected)
{
    if (rejected[blockIdx.x])
    {
        const auto task = tasks[blockIdx.x];
        fused_task_compute(factors + task.factor_offset, input, task, true);
    }
}

struct TaskSet
{
    Buffer<Job> jobs;
    Buffer<uint16_t> codes;
    Buffer<unsigned> rejected;
    std::shared_ptr<const Lookup> table;
    unsigned count = 0, tiles = 0;

    TaskSet(const std::vector<Job>& descriptors, size_t order_sum, unsigned max_tiles)
        : table(lookup()), count(unsigned(descriptors.size())), tiles(selected_tiles(max_tiles))
    {
        jobs.upload(descriptors);
        codes.allocate(order_sum);
        rejected.allocate(count);
    }

    size_t bytes() const { return jobs.bytes + codes.bytes + rejected.bytes; }

    void enqueue(REAL* factors,
        const REAL* input,
        const FusedTaskDescriptor* tasks,
        unsigned fused_threads,
        cudaStream_t stream)
    {
        gpu_check(cudaMemsetAsync(rejected.ptr, 0, rejected.bytes, stream), "clear diagonal scale status");
        const dim3 grid(count, tiles);
        recover_rows<<<grid, threads, 0, stream>>>(input, tasks, jobs.ptr, codes.ptr, rejected.ptr, table->heads.ptr,
            table->pairs.ptr);
        gpu_check(cudaGetLastError(), "recover diagonal scales from actual compact input");
        verify_inputs<<<grid, threads, 0, stream>>>(input, tasks, jobs.ptr, codes.ptr, rejected.ptr);
        gpu_check(cudaGetLastError(), "verify all compact input bytes before reuse");
        write_factors<<<grid, threads, 0, stream>>>(factors, tasks, jobs.ptr, codes.ptr, rejected.ptr,
            table->reciprocals.ptr);
        gpu_check(cudaGetLastError(), "reconstruct all verified factors");
        fallback<<<count, fused_threads, 0, stream>>>(factors, input, tasks, rejected.ptr);
        gpu_check(cudaGetLastError(), "factor unverified matrices with Fused");
    }

    void report() const
    {
        std::vector<unsigned> states(count);
        gpu_check(cudaMemcpy(states.data(), rejected.ptr, rejected.bytes, cudaMemcpyDeviceToHost),
            "read diagonal scale status");
        const size_t verified = std::count(states.begin(), states.end(), 0u);
        std::fprintf(stderr, "SCALE_REUSE,%zu,%u,%zu\n", verified, count, table->bytes());
    }
};

} // namespace diagonal_scale
