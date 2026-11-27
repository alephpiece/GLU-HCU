// Single-copy right-looking LU on nested sparse elimination fronts.
// The first pivot's Mapped indices locate every sparse entry of the rectangle.
// Diagonals and raw L stay in the usual factor representation for final normalization.
constexpr unsigned front_tile = 16;

template <bool Pack, class Slot = unsigned>
__global__ void mapped_front_copy(REAL* sparse,
    REAL* dense,
    unsigned rows,
    unsigned cols,
    unsigned diagonal,
    const Slot* destinations,
    const unsigned* pivots)
{
    const unsigned position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= rows * cols)
        return;
    const unsigned r = position % rows, c = position / rows;
    unsigned slot;
    if (c == 0)
        slot = diagonal + r;
    else if (r == 0)
        slot = pivots[c - 1];
    else
        slot = mapped_position(destinations, (c - 1) * (rows - 1) + r - 1, pivots[c - 1]);
    if constexpr (Pack)
        dense[position] = sparse[slot];
    else
        sparse[slot] = dense[position];
}

template <bool Perturb> __device__ inline REAL mapped_front_divisor(REAL value, REAL threshold)
{
    if constexpr (Perturb)
        if (fabs(value) < threshold)
            return threshold;
    return value;
}

// Each destination has a unique writer throughout this kernel.
__device__ inline REAL mapped_front_subtract(REAL value, REAL factor, REAL pivot, REAL* destination)
{
#if defined(GLU_USE_HIP)
    // HIP headers may implement the rounded multiply with the ordinary operator.
#pragma clang fp contract(off)
#endif
    if (factor == REAL(0))
        return value;
#if defined(GLU_USE_FP32)
    const float product = __fmul_rn(-factor, pivot);
#if !defined(GLU_USE_HIP)
    // CUDA global atomicAdd flushes FP32 inputs and results to signed zero.
    float sum;
    asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(sum) : "f"(value), "f"(product));
    return sum;
#elif defined(__gfx936__)
    // Multiplication by one is exact. This instruction flushes subnormals
    // on the add, matching gfx936 global atomicAdd.
    float sum;
    asm("v_mad_f32 %0, 1.0, %1, %2" : "=v"(sum) : "v"(value), "v"(product));
    return sum;
#else
    // Use this HIP target's actual atomic arithmetic on the private destination.
    *destination = value;
    atomicAdd(destination, product);
    return *destination;
#endif
#else
    return value + __dmul_rn(-factor, pivot);
#endif
}

template <bool Perturb>
__global__ void mapped_front_diagonal(REAL* dense, unsigned ld, unsigned offset, unsigned steps, REAL threshold)
{
    __shared__ REAL tile[front_tile][front_tile + 1], normalized[front_tile];
    const unsigned r = threadIdx.x % front_tile, c = threadIdx.x / front_tile;
    const bool active = r < steps && c < steps;
    const unsigned slot = offset + r + (offset + c) * ld;
    if (active)
        tile[c][r] = dense[slot];
    __syncthreads();
    for (unsigned k = 0; k < steps; ++k)
    {
        if (c == k && r > k && r < steps)
            normalized[r] = tile[k][r] / mapped_front_divisor<Perturb>(tile[k][k], threshold);
        __syncthreads();
        if (active && r > k && c > k)
            tile[c][r] = mapped_front_subtract(tile[c][r], normalized[r], tile[c][k], dense + slot);
        __syncthreads();
    }
    if (active)
        dense[slot] = tile[c][r];
}

// Each group of front_tile lanes solves one vertical or horizontal edge.
// One lane holds each panel element and broadcasts it when it becomes the pivot.
template <bool Perturb>
__global__ void mapped_front_edges(REAL* dense,
    REAL* lower,
    unsigned rows,
    unsigned cols,
    unsigned offset,
    unsigned steps,
    REAL threshold)
{
    __shared__ REAL tile[front_tile][front_tile + 1];
    const unsigned r = threadIdx.x % front_tile, c = threadIdx.x / front_tile;
    if (r < steps && c < steps)
        tile[c][r] = dense[(offset + c) * rows + offset + r];
    __syncthreads();
    if (r < steps && c < steps && r > c)
        tile[c][r] /= mapped_front_divisor<Perturb>(tile[c][c], threshold);
    __syncthreads();
    const unsigned lane = r;
    const unsigned outer = offset + steps + blockIdx.x * (blockDim.x / front_tile) + c;
    const bool vertical = blockIdx.y == 0;
    const bool active = outer < (vertical ? rows : cols) && lane < steps;
    const unsigned slot = vertical ? outer + (offset + lane) * rows : offset + lane + outer * rows;
    REAL value = active ? dense[slot] : REAL(0);
    for (unsigned k = 0; k < steps; ++k)
    {
        REAL current = value;
        if (vertical && active && lane == k)
        {
            current /= mapped_front_divisor<Perturb>(tile[k][k], threshold);
            lower[k * rows + outer] = current;
        }
        // Inactive groups and lanes also participate in the shuffle.
#if defined(GLU_USE_HIP)
        const REAL pivot = __shfl(current, k, front_tile);
#else
        const REAL pivot = __shfl_sync(0xffffffffu, current, k, front_tile);
#endif
        if (active && lane > k)
            value = mapped_front_subtract(value, vertical ? pivot : tile[k][lane],
                vertical ? tile[lane][k] : pivot, dense + slot);
    }
    if (active)
        dense[slot] = value;
}

__global__ void
mapped_front_schur(REAL* dense, const REAL* lower, unsigned rows, unsigned cols, unsigned offset, unsigned steps)
{
    __shared__ REAL l[front_tile][front_tile + 1], u[front_tile][front_tile + 1];
    const unsigned x = threadIdx.x % front_tile, y = threadIdx.x / front_tile;
    const unsigned r = offset + steps + blockIdx.x * front_tile + x;
    const unsigned c = offset + steps + blockIdx.y * front_tile + y;
    if (r < rows && y < steps)
        l[y][x] = lower[y * rows + r];
    if (c < cols && x < steps)
        u[y][x] = dense[c * rows + offset + x];
    __syncthreads();
    if (r < rows && c < cols)
    {
        REAL value = dense[c * rows + r];
        for (unsigned k = 0; k < steps; ++k)
            value = mapped_front_subtract(value, l[k][x], u[y][k], dense + c * rows + r);
        dense[c * rows + r] = value;
    }
}

struct MappedFront
{
    unsigned steps = 0, rows = 0, cols = 0, diagonal = 0, first_target = 0;
    uint64_t map_begin = 0;

    unsigned numeric_launches() const
    {
        unsigned launches = 2; // Gather and scatter the rectangle.
        for (unsigned offset = 0; offset < steps; offset += front_tile)
        {
            const unsigned end = std::min(offset + front_tile, steps);
            ++launches; // Factor the diagonal tile.
            if (end < rows || end < cols)
                ++launches;
            if (end < rows && end < cols)
                ++launches;
        }
        return launches;
    }
};

struct MappedFrontPlan
{
    // Indexed by the first dependency level; empty for matrices without a front.
    std::vector<MappedFront> starting_at;
    size_t workspace_elements = 0;
    struct SplitLevel
    {
        bool split = false;
        std::vector<MappedFront> fronts;
        UpdateRange sparse{};
    };
    std::vector<SplitLevel> split_levels;
};

constexpr unsigned front_minimum_steps = 32;
constexpr uint64_t front_maximum_elements = 1024 * 1024;

bool mapped_front_shape(const Symbolic_Matrix& a, unsigned root, unsigned rows, unsigned cols)
{
    if (rows < front_minimum_steps || cols < front_minimum_steps ||
        uint64_t(rows) * cols > front_maximum_elements)
        return false;
    const auto lower = a.sym_r_idx.begin() + a.l_col_ptr[root];
    const auto upper = a.csr_c_idx.begin() + a.csr_diag_ptr[root];
    const auto nonincreasing = [](unsigned first, unsigned second) { return first >= second; };
    // Packing and scattering require distinct rows and columns.
    return std::adjacent_find(lower, lower + rows, nonincreasing) == lower + rows &&
           std::adjacent_find(upper, upper + cols, nonincreasing) == upper + cols;
}

// Each pivot is the next row and column of the previous front. Equality of
// both tails keeps the remaining work inside the first pivot's rectangle.
unsigned nested_front_successor(const Symbolic_Matrix& a, unsigned current)
{
    const auto lo = a.sym_r_idx.begin() + a.l_col_ptr[current] + 1;
    const auto lo_end = a.sym_r_idx.begin() + a.sym_c_ptr[current + 1];
    const auto up = a.csr_c_idx.begin() + a.csr_diag_ptr[current] + 1;
    const auto up_end = a.csr_c_idx.begin() + a.csr_r_ptr[current + 1];
    if (lo == lo_end || up == up_end || *lo != *up)
        return a.n;
    const unsigned next = *lo;
    if (next >= a.n)
        return a.n;
    if (std::equal(lo + 1, lo_end, a.sym_r_idx.begin() + a.l_col_ptr[next] + 1,
            a.sym_r_idx.begin() + a.sym_c_ptr[next + 1]) &&
        std::equal(up + 1, up_end, a.csr_c_idx.begin() + a.csr_diag_ptr[next] + 1,
            a.csr_c_idx.begin() + a.csr_r_ptr[next + 1]))
        return next;
    return a.n;
}

void add_split_mapped_levels(MappedFrontPlan& plan, const Symbolic_Matrix& a, const std::vector<uint64_t>& bases,
    std::vector<int>& mapped_order)
{
    // A separate front entails packing and several launches. Small fronts
    // can leave almost all of the original sparse launches in place.
#if defined(GLU_USE_HIP)
    constexpr uint64_t minimum_products = 5000000;
#else
    constexpr uint64_t minimum_products = 1000000;
#endif
    // At most n columns can be appended, and update bounds use signed indices.
    if (uint64_t(a.n) * 2 > uint64_t(std::numeric_limits<int>::max()))
        return;
    const auto& levels = a.level_ptr;
    const auto& columns = a.level_idx;
    const auto eligible = [&](unsigned lev) {
        // These levels use raw L with a common final normalization.
        const int width = levels[lev + 1] - levels[lev];
        return width >= 2 && width <= 16;
    };
    std::vector<unsigned> source_level(a.n), member(a.n, a.n), chain;
    std::vector<bool> assigned(a.n);
    for (unsigned lev = 0; lev + 1 < levels.size(); ++lev)
        for (int i = levels[lev]; i < levels[lev + 1]; ++i)
            source_level[unsigned(columns[i])] = lev;

    for (unsigned lev = 0; lev + 1 < levels.size(); ++lev)
    {
        if (!eligible(lev))
            continue;
        for (int i = levels[lev]; i < levels[lev + 1]; ++i)
        {
            const unsigned root = unsigned(columns[i]);
            const unsigned rows = a.sym_c_ptr[root + 1] - a.l_col_ptr[root];
            const unsigned cols = a.csr_r_ptr[root + 1] - a.csr_diag_ptr[root];
            if (assigned[root] || !mapped_front_shape(a, root, rows, cols))
                continue;
            chain.assign(1, root);
            member[root] = root;
            uint64_t products = bases[root + 1] - bases[root];
            for (unsigned current = root;;)
            {
                const unsigned next = nested_front_successor(a, current);
                if (next == a.n || assigned[next] || source_level[next] != source_level[current] + 1 ||
                    !eligible(source_level[next]))
                    break;
                // The next pivot needs its complete row and column. Move it to
                // this level only if earlier levels or this front provide both.
                const auto pending = [&](unsigned source) {
                    return source_level[source] >= lev && member[source] != root;
                };
                if (std::any_of(a.sym_r_idx.begin() + a.sym_c_ptr[next], a.sym_r_idx.begin() + a.l_col_ptr[next],
                        pending) ||
                    std::any_of(a.csr_c_idx.begin() + a.csr_r_ptr[next], a.csr_c_idx.begin() + a.csr_diag_ptr[next],
                        pending))
                    break;
                member[next] = root;
                chain.push_back(next);
                products += bases[next + 1] - bases[next];
                current = next;
            }
            if (chain.size() < front_minimum_steps || products < minimum_products)
                continue;
            if (plan.split_levels.empty())
                plan.split_levels.resize(levels.size() - 1);
            plan.split_levels[lev].fronts.push_back(
                {unsigned(chain.size()), rows, cols, a.l_col_ptr[root], a.csr_diag_ptr[root] + 1, bases[root]});
            for (unsigned source : chain)
            {
                assigned[source] = true;
                plan.split_levels[source_level[source]].split = true;
            }
            plan.workspace_elements = std::max(plan.workspace_elements, size_t(rows) * (cols + front_tile));
        }
    }
    for (unsigned lev = 0; lev < plan.split_levels.size(); ++lev)
    {
        auto& split = plan.split_levels[lev];
        if (!split.split)
            continue;
        // Keep the original order for other paths. Append only the sparse
        // remainder of levels that would otherwise require multiple launches.
        unsigned begin = unsigned(levels[lev]), end = unsigned(levels[lev + 1]);
        while (begin < end && assigned[unsigned(columns[begin])])
            ++begin;
        while (end > begin && assigned[unsigned(columns[end - 1])])
            --end;
        const auto is_assigned = [&](int column) { return assigned[unsigned(column)]; };
        if (std::none_of(columns.begin() + begin, columns.begin() + end, is_assigned))
            split.sparse = {begin, end};
        else
        {
            const unsigned first = unsigned(mapped_order.size());
            for (unsigned i = begin; i < end; ++i)
                if (!is_assigned(columns[i]))
                    mapped_order.push_back(columns[i]);
            split.sparse = {first, unsigned(mapped_order.size())};
        }
    }
}

MappedFrontPlan make_mapped_front_plan(
    const Symbolic_Matrix& a, const std::vector<uint64_t>& bases, std::vector<int>& mapped_order)
{
    const auto& levels = a.level_ptr;
    const auto& columns = a.level_idx;
    MappedFrontPlan plan;
    constexpr unsigned minimum_single_source_steps = 8;
    for (size_t lev = 0; lev + 1 < levels.size();)
    {
        if (levels[lev + 1] - levels[lev] != 1)
        {
            ++lev;
            continue;
        }
        const unsigned c = unsigned(columns[levels[lev]]);
        const unsigned rows = a.sym_c_ptr[c + 1] - a.l_col_ptr[c];
        const unsigned cols = a.csr_r_ptr[c + 1] - a.csr_diag_ptr[c];
        if (!mapped_front_shape(a, c, rows, cols))
        {
            ++lev;
            continue;
        }
        size_t end = lev + 1;
        for (; end + 1 < levels.size() && levels[end + 1] - levels[end] == 1; ++end)
        {
            const unsigned prior = unsigned(columns[levels[end - 1]]);
            const unsigned next = unsigned(columns[levels[end]]);
            if (nested_front_successor(a, prior) != next)
                break;
        }
        if (end - lev >= minimum_single_source_steps)
        {
            if (plan.starting_at.empty())
                plan.starting_at.resize(levels.size() - 1);
            const unsigned steps = unsigned(end - lev);
            plan.starting_at[lev] = {steps, rows, cols, a.l_col_ptr[c], a.csr_diag_ptr[c] + 1, bases[c]};
            plan.workspace_elements = std::max(plan.workspace_elements, size_t(rows) * (cols + front_tile));
        }
        lev = end;
    }
    if constexpr (!normalize_mapped_levels)
        add_split_mapped_levels(plan, a, bases, mapped_order);
    return plan;
}

template <bool Perturb, class Slot = unsigned>
void enqueue_mapped_front(const MappedFront& front,
    REAL* sparse,
    REAL* dense,
    const Slot* destination_map,
    const unsigned* pivot_map,
    REAL threshold,
    cudaStream_t stream)
{
    const unsigned rows = front.rows, cols = front.cols;
    REAL* lower = dense + size_t(rows) * cols;
    const unsigned blocks = (rows * cols + 255) / 256;
    auto* destinations = destination_map + front.map_begin;
    auto* pivots = pivot_map + front.first_target;
    mapped_front_copy<true>
        <<<blocks, 256, 0, stream>>>(sparse, dense, rows, cols, front.diagonal, destinations, pivots);
    gpu_check(cudaGetLastError(), "gather mapped front");
    for (unsigned offset = 0; offset < front.steps; offset += front_tile)
    {
        const unsigned steps = std::min(front_tile, front.steps - offset);
        const unsigned remaining_rows = rows - offset - steps, remaining_cols = cols - offset - steps;
        mapped_front_diagonal<Perturb><<<1, 256, 0, stream>>>(dense, rows, offset, steps, threshold);
        gpu_check(cudaGetLastError(), "factor mapped front diagonal");
        if (remaining_rows || remaining_cols)
        {
            const unsigned edges = std::max(remaining_rows, remaining_cols);
            constexpr unsigned edges_per_block = 256 / front_tile;
            mapped_front_edges<Perturb>
                <<<dim3((edges + edges_per_block - 1) / edges_per_block, 2), 256, 0, stream>>>(
                    dense, lower, rows, cols, offset, steps, threshold);
            gpu_check(cudaGetLastError(), "factor mapped front edges");
        }
        if (remaining_rows && remaining_cols)
        {
            const dim3 grid((remaining_rows + front_tile - 1) / front_tile,
                (remaining_cols + front_tile - 1) / front_tile);
            mapped_front_schur<<<grid, 256, 0, stream>>>(dense, lower, rows, cols, offset, steps);
            gpu_check(cudaGetLastError(), "update mapped front");
        }
    }
    mapped_front_copy<false>
        <<<blocks, 256, 0, stream>>>(sparse, dense, rows, cols, front.diagonal, destinations, pivots);
    gpu_check(cudaGetLastError(), "scatter mapped front");
}
