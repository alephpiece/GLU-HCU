// Requires update-plan.h. D holes are structural absences, never stores.
__device__ void scalar_dense_outputs(REAL* values,
    const DenseUpdateTile& tile,
    const unsigned* indices,
    unsigned copy,
    unsigned stride,
    unsigned lane,
    unsigned lanes)
{
    for (unsigned output = lane; output < 256; output += lanes)
    {
        const unsigned r = tile.row_begin + output % 16, c = tile.col_begin + output / 16;
        if (r >= tile.nrows || c >= tile.ncols)
            continue;
        const unsigned dst = indices[tile.dst_begin + c * tile.nrows + r];
        if (dst == UINT32_MAX)
            continue;
        REAL acc = values[size_t(dst) * stride + copy];
        for (unsigned q = 0; q < tile.length; ++q)
        {
            const REAL factor = values[size_t(indices[tile.l_begin + q * tile.nrows + r]) * stride + copy];
            if (factor != REAL(0))
                acc -= factor * values[size_t(indices[tile.u_begin + q * tile.ncols + c]) * stride + copy];
        }
        values[size_t(dst) * stride + copy] = acc;
    }
}

#if defined(__gfx936__)
// REAL selects native f32/f64 operands and accumulation; no TF32 conversion.
using DenseMmacVector = REAL __attribute__((ext_vector_type(4)));
__device__ __forceinline__ DenseMmacVector dense_mmac(REAL a, REAL b, DenseMmacVector c)
{
#if defined(GLU_USE_FP32)
    asm volatile("s_nop 4\n\tv_mmac_16x16x4_f32 %0, %1, %2, %0\n\ts_nop 2" : "+v"(c) : "v"(a), "v"(b));
#else
    asm volatile("s_nop 4\n\tv_mmac_16x16x4_f64 %0, %1, %2, %0\n\ts_nop 2" : "+v"(c) : "v"(a), "v"(b));
#endif
    return c;
}
#endif

// A 256-thread block handles one tile for up to four copies, 64 lanes per copy.
// On gfx936 each copy occupies a whole wave; other targets use scalar outputs.
__global__ void panel_tiled_update(REAL* values,
    unsigned copies,
    unsigned stride,
    const DenseUpdateTile* tiles,
    const unsigned* indices,
    unsigned begin)
{
    const unsigned lane = threadIdx.x % 64, copy = blockIdx.y * 4 + threadIdx.x / 64;
    if (copy >= copies)
        return;
    const auto tile = tiles[begin + blockIdx.x];

#if defined(__gfx936__)
    // Each lane owns four accumulators in row lane%16, at columns
    // lane/16 + 4*part within this view of the rectangle.
    const unsigned r = tile.row_begin + lane % 16, cbase = tile.col_begin;
    DenseMmacVector acc{};
    bool invalid = false;
    for (unsigned part = 0; part < 4; ++part)
    {
        const unsigned c = cbase + lane / 16 + 4 * part;
        if (r < tile.nrows && c < tile.ncols)
        {
            const unsigned dst = indices[tile.dst_begin + c * tile.nrows + r];
            if (dst != UINT32_MAX)
                acc[part] = values[size_t(dst) * stride + copy];
        }
    }

    // MMAC advances four reduction terms: lane/16 selects K, while lane%16
    // selects the L row and U column. Rectangle and K tails supply zeros.
    for (unsigned first = 0; first < tile.length; first += 4)
    {
        const unsigned q = first + lane / 16, c = cbase + lane % 16;
        const REAL l = (r < tile.nrows && q < tile.length)
                           ? values[size_t(indices[tile.l_begin + q * tile.nrows + r]) * stride + copy]
                           : REAL(0);
        const REAL u = (c < tile.ncols && q < tile.length)
                           ? values[size_t(indices[tile.u_begin + q * tile.ncols + c]) * stride + copy]
                           : REAL(0);
        invalid |= !isfinite(l) || !isfinite(u);
        acc = dense_mmac(-l, u, acc);
    }

    // Preserve the scalar zero-factor rule if a tile contains nonfinite operands.
    // No output has been stored yet: the whole wave can recompute from the
    // original destination values before either path writes its results.
    if (__any(invalid))
    {
        scalar_dense_outputs(values, tile, indices, copy, stride, lane, 64);
        return;
    }

    for (unsigned part = 0; part < 4; ++part)
    {
        const unsigned c = cbase + lane / 16 + 4 * part;
        if (r < tile.nrows && c < tile.ncols)
        {
            const unsigned dst = indices[tile.dst_begin + c * tile.nrows + r];
            if (dst != UINT32_MAX)
                values[size_t(dst) * stride + copy] = acc[part];
        }
    }
#else
    // Scalar fallback on targets without the selected native instruction path.
    scalar_dense_outputs(values, tile, indices, copy, stride, lane, 64);
#endif
}
