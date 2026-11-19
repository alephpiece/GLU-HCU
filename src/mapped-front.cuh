// Single-copy right-looking LU on a nested sequence of sparse elimination fronts.
// The first pivot's Mapped indices locate every sparse entry of the rectangle.
// Diagonals and raw L stay in the usual factor representation for final normalization.
constexpr unsigned front_tile = 16;

template <bool Pack>
__global__ void mapped_front_copy(REAL* sparse,
    REAL* dense,
    unsigned rows,
    unsigned cols,
    unsigned diagonal,
    const unsigned* destinations,
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
        slot = destinations[(c - 1) * (rows - 1) + r - 1];
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

// Match the single-source sparse update's rounded product before accumulation.
__device__ inline REAL mapped_front_subtract(REAL value, REAL factor, REAL pivot)
{
    if (factor == REAL(0))
        return value;
#if defined(GLU_USE_FP32)
    return value + __fmul_rn(-factor, pivot);
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
            tile[c][r] = mapped_front_subtract(tile[c][r], normalized[r], tile[c][k]);
        __syncthreads();
    }
    if (active)
        dense[slot] = tile[c][r];
}

// Vertical edge threads factor a row against the diagonal tile and retain
// its normalized L entries for Schur updates. Horizontal threads factor a column.
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
    const unsigned outer = offset + steps + blockIdx.x * blockDim.x + threadIdx.x;
    const bool vertical = blockIdx.y == 0;
    if (outer >= (vertical ? rows : cols))
        return;
    REAL panel[front_tile];
    for (unsigned k = 0; k < steps; ++k)
        panel[k] = vertical ? dense[outer + (offset + k) * rows] : dense[offset + k + outer * rows];
    for (unsigned k = 0; k < steps; ++k)
    {
        if (vertical)
        {
            const REAL factor = panel[k] / mapped_front_divisor<Perturb>(tile[k][k], threshold);
            lower[k * rows + outer] = factor;
            for (unsigned j = k + 1; j < steps; ++j)
                panel[j] = mapped_front_subtract(panel[j], factor, tile[j][k]);
        }
        else
            for (unsigned i = k + 1; i < steps; ++i)
                panel[i] = mapped_front_subtract(panel[i], tile[k][i], panel[k]);
    }
    for (unsigned k = 0; k < steps; ++k)
    {
        const unsigned slot = vertical ? outer + (offset + k) * rows : offset + k + outer * rows;
        dense[slot] = panel[k];
    }
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
            value = mapped_front_subtract(value, l[k][x], u[y][k]);
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
};

MappedFrontPlan make_mapped_front_plan(const Symbolic_Matrix& a, const std::vector<uint64_t>& bases)
{
    const auto& levels = a.level_ptr;
    const auto& columns = a.level_idx;
    MappedFrontPlan plan;
    // Amortize packing over at least two panels and bound per-task temporary storage.
    constexpr unsigned minimum_steps = 32;
    constexpr uint64_t maximum_elements = 1024 * 1024;
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
        if (rows < minimum_steps || cols < minimum_steps || uint64_t(rows) * cols > maximum_elements)
        {
            ++lev;
            continue;
        }
        // Each pivot must be the next row and column of the previous front.
        // Equality of both remaining tails proves that the exterior rows and
        // columns also belong to the rectangle gathered at the first pivot.
        size_t end = lev + 1;
        for (; end + 1 < levels.size() && levels[end + 1] - levels[end] == 1; ++end)
        {
            const unsigned prior = unsigned(columns[levels[end - 1]]);
            const unsigned next = unsigned(columns[levels[end]]);
            auto lo = a.sym_r_idx.begin() + a.l_col_ptr[prior] + 1;
            auto lo_end = a.sym_r_idx.begin() + a.sym_c_ptr[prior + 1];
            auto up = a.csr_c_idx.begin() + a.csr_diag_ptr[prior] + 1;
            auto up_end = a.csr_c_idx.begin() + a.csr_r_ptr[prior + 1];
            if (lo == lo_end || up == up_end || *lo != next || *up != next ||
                !std::equal(lo + 1, lo_end, a.sym_r_idx.begin() + a.l_col_ptr[next] + 1,
                    a.sym_r_idx.begin() + a.sym_c_ptr[next + 1]) ||
                !std::equal(up + 1, up_end, a.csr_c_idx.begin() + a.csr_diag_ptr[next] + 1,
                    a.csr_c_idx.begin() + a.csr_r_ptr[next + 1]))
                break;
        }
        if (end - lev >= minimum_steps)
        {
            if (plan.starting_at.empty())
                plan.starting_at.resize(levels.size() - 1);
            const unsigned steps = unsigned(end - lev);
            plan.starting_at[lev] = {steps, rows, cols, a.l_col_ptr[c], a.csr_diag_ptr[c] + 1, bases[c]};
            plan.workspace_elements = std::max(plan.workspace_elements, size_t(rows) * (cols + front_tile));
        }
        lev = end;
    }
    return plan;
}

template <bool Perturb>
void enqueue_mapped_front(const MappedFront& front,
    REAL* sparse,
    REAL* dense,
    const unsigned* destination_map,
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
            mapped_front_edges<Perturb>
                <<<dim3((edges + 255) / 256, 2), 256, 0, stream>>>(dense, lower, rows, cols, offset, steps, threshold);
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
