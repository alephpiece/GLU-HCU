// All tasks share one update launch at each active dependency level.
// Write L once per level when reusing it across targets offsets the extra launches.
// With deferred writeback, updates divide their source values on demand.
#if defined(GLU_USE_HIP) || defined(GLU_USE_FP32)
constexpr bool defer_set_normalization = true;
#else
constexpr bool defer_set_normalization = false;
#endif

// Each source-target update owns its L-column lanes. Independent sources may update
// the same factor slot, as on the single-copy right-looking Mapped path.
struct MappedSetTask
{
    size_t factors, input;
    unsigned compact_count;
    const unsigned *positions, *col, *diag, *csr_diag, *destinations, *pivots;
    const uint64_t* bases;
    const uint16_t* deltas;
    REAL threshold;
};

struct MappedSetColumn
{
    unsigned task, source;
};
struct MappedSetUpdate
{
    unsigned task, source, target, count;
};
struct MappedSetStage
{
    UpdateRange columns, updates;
};

__global__ void scatter_mapped_set(REAL* factors, const REAL* input, const MappedSetTask* tasks, size_t task_count)
{
    for (size_t job = blockIdx.y; job < task_count; job += gridDim.y)
    {
        const auto task = tasks[job];
        for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < task.compact_count;
             i += size_t(blockDim.x) * gridDim.x)
            factors[task.factors + task.positions[i]] = input[task.input + i];
    }
}

template <bool Perturb>
__global__ void normalize_mapped_set(REAL* factors,
    const MappedSetTask* tasks,
    const MappedSetColumn* columns,
    unsigned begin,
    unsigned end)
{
    const size_t j = size_t(begin) + size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (j >= end)
        return;
    const auto column = columns[j];
    const auto task = tasks[column.task];
    REAL* values = factors + task.factors;
    const unsigned diagonal = task.diag[column.source];
    REAL divisor;
    if constexpr (Perturb)
    {
        divisor = REAL(0);
        if (threadIdx.x % 32 == 0)
        {
            divisor = values[diagonal];
            if (fabs(divisor) < task.threshold)
                values[diagonal] = divisor = task.threshold;
        }
        divisor = gpuShuffle32(divisor, 0);
    }
    else
        divisor = values[diagonal];
    for (size_t i = size_t(diagonal) + 1 + threadIdx.x % 32; i < task.col[column.source + 1]; i += 32)
        values[i] /= divisor;
}

template <class Slot>
__device__ inline void update_mapped_target(REAL* values,
    const Slot* destinations,
    uint64_t base,
    unsigned lower,
    unsigned count,
    unsigned pivot_position,
    REAL divisor)
{
    const REAL pivot = values[pivot_position];
    for (size_t i = threadIdx.x % 32; i < count; i += 32)
    {
        REAL factor = values[lower + i];
        if constexpr (defer_set_normalization)
            factor /= divisor;
        if (factor != REAL(0))
            atomicAdd(values + mapped_position(destinations, base + i, pivot_position), -factor * pivot);
    }
}

template <bool Perturb>
__global__ void
update_mapped_set(REAL* factors, const MappedSetTask* tasks, const MappedSetUpdate* updates, unsigned begin)
{
    const auto entry = updates[size_t(begin) + blockIdx.x];
    const unsigned group = threadIdx.x / 32;
    if (group >= entry.count)
        return;
    const auto task = tasks[entry.task];
    REAL* values = factors + task.factors;
    const unsigned target = entry.target + group;
    const unsigned diagonal = task.diag[entry.source], lower = diagonal + 1;
    const unsigned count = task.col[entry.source + 1] - lower;
    const uint64_t base = task.bases[entry.source] + uint64_t(target) * count;
    const unsigned pivot = task.pivots[task.csr_diag[entry.source] + 1 + target];
    REAL divisor = REAL(1);
    if constexpr (defer_set_normalization)
    {
        // Updates may share the source pivot. Only normalization writes it.
        divisor = values[diagonal];
        if constexpr (Perturb)
            if (fabs(divisor) < task.threshold)
                divisor = task.threshold;
    }
    if (task.deltas)
        update_mapped_target(values, task.deltas, base, lower, count, pivot, divisor);
    else
        update_mapped_target(values, task.destinations, base, lower, count, pivot, divisor);
}

struct MappedSetPlan
{
    dim3 scatter_grid;
    Buffer<MappedSetTask> tasks;
    Buffer<MappedSetColumn> columns;
    Buffer<MappedSetUpdate> updates;
    std::vector<MappedSetStage> stages;
    cudaGraphExec_t graph = nullptr;

    ~MappedSetPlan()
    {
        if (graph)
            cudaGraphExecDestroy(graph);
    }

    template <class Templates>
    void build(const Templates& templates,
        const std::vector<size_t>& indices,
        const std::vector<size_t>& factors,
        const std::vector<size_t>& compact)
    {
        const auto offset = [](size_t size)
        {
            if (size > std::numeric_limits<unsigned>::max())
                throw std::overflow_error("mapped set index overflow");
            return unsigned(size);
        };
        size_t depth = 0;
        unsigned scatter_blocks = 1;
        std::vector<MappedSetTask> task_specs;
        for (size_t job = 0; job < indices.size(); ++job)
        {
            const auto& a = *templates[indices[job]];
            task_specs.push_back({factors[job], compact[job], a.compact_elements, a.input_positions.ptr, a.col.ptr,
                a.diag.ptr, a.csr_diag.ptr, a.map_destinations.ptr, a.map_pivots.ptr, a.map_bases.ptr, a.map_deltas.ptr,
                a.threshold});
            depth = std::max(depth, a.levels.size() - 1);
            scatter_blocks =
                std::max(scatter_blocks, (a.compact_elements / 256) + unsigned(a.compact_elements % 256 != 0));
        }
        const unsigned limit = templates[indices.front()]->max_grid_x;
        scatter_grid = dim3(std::min(scatter_blocks, limit),
            unsigned(std::min(indices.size(), size_t(templates[indices.front()]->max_batch))));
        std::vector<MappedSetColumn> column_specs;
        std::vector<MappedSetUpdate> update_specs;
        constexpr unsigned targets_per_block = 256 / 32;
        for (size_t level = 0; level < depth; ++level)
        {
            const unsigned cb = offset(column_specs.size()), ub = offset(update_specs.size());
            for (size_t job = 0; job < indices.size(); ++job)
            {
                const auto& a = *templates[indices[job]];
                if (level + 1 >= a.levels.size())
                    continue;
                for (int pos = a.levels[level]; pos < a.levels[level + 1]; ++pos)
                {
                    const unsigned source = unsigned(a.columns[pos]);
                    column_specs.push_back({unsigned(job), source});
                    const uint64_t count = a.target_prefix[pos + 1] - a.target_prefix[pos];
                    for (uint64_t q = 0; q < count; q += targets_per_block)
                        update_specs.push_back({unsigned(job), source, unsigned(q),
                            unsigned(std::min(uint64_t(targets_per_block), count - q))});
                }
            }
            const unsigned ce = offset(column_specs.size()), ue = offset(update_specs.size());
            const uint64_t normalized_columns = defer_set_normalization ? ce : ce - cb;
            if (ue - ub > limit || (normalized_columns + 7) / 8 > limit)
                throw std::invalid_argument("mapped set level exceeds device grid limit");
            stages.push_back({{cb, ce}, {ub, ue}});
        }
        tasks.upload(task_specs);
        columns.upload(column_specs);
        updates.upload(update_specs);
    }

    template <bool Perturb> void normalize(REAL* values, cudaStream_t stream, UpdateRange range) const
    {
        const unsigned blocks = unsigned((uint64_t(range.end - range.begin) + 7) / 8);
        if (blocks)
        {
            normalize_mapped_set<Perturb>
                <<<blocks, 256, 0, stream>>>(values, tasks.ptr, columns.ptr, range.begin, range.end);
            gpu_check(cudaGetLastError(), "normalize mapped task set");
        }
    }

    template <bool Perturb> void enqueue(REAL* values, cudaStream_t stream) const
    {
        for (const auto& stage : stages)
        {
            if constexpr (!defer_set_normalization)
                normalize<Perturb>(values, stream, stage.columns);
            const unsigned blocks = stage.updates.end - stage.updates.begin;
            if (blocks)
            {
                update_mapped_set<Perturb>
                    <<<blocks, 256, 0, stream>>>(values, tasks.ptr, updates.ptr, stage.updates.begin);
                gpu_check(cudaGetLastError(), "update mapped task set");
            }
        }
        if constexpr (defer_set_normalization)
            normalize<Perturb>(values, stream, {0, stages.back().columns.end});
    }
};
