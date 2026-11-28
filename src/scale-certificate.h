#pragma once

#include "symbolic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace diagonal_scale
{

constexpr unsigned denominator = 1024;
constexpr unsigned choices = 256;
constexpr unsigned first_product = denominator * denominator;
constexpr unsigned last_product = (denominator + choices - 1) * (denominator + choices - 1);
constexpr unsigned missing = std::numeric_limits<unsigned>::max();
constexpr unsigned max_order = 1u << 15;

struct alignas(uint32_t) FactorCoordinate
{
    uint16_t row, column;
};
static_assert(sizeof(FactorCoordinate) == sizeof(uint32_t));

struct ProductTable
{
    // A product has up to ten distinct ordered row/column pairs in this grid.
    std::vector<uint32_t> heads = std::vector<uint32_t>(last_product - first_product + 1);
    std::vector<uint16_t> pairs;
    std::vector<REAL> reciprocals = std::vector<REAL>(choices);

    ProductTable()
    {
        for (unsigned left = 0; left < choices; ++left)
            for (unsigned right = 0; right < choices; ++right)
                ++heads[(denominator + left) * (denominator + right) - first_product];
        unsigned offset = 0;
        for (uint32_t& head : heads)
        {
            if (head > 15)
                throw std::logic_error("scale product table width");
            const unsigned count = head;
            head = (offset << 4) | count;
            offset += count;
        }
        pairs.resize(offset);
        std::vector<uint8_t> used(heads.size());
        for (unsigned left = 0; left < choices; ++left)
        {
            reciprocals[left] = REAL(denominator) / REAL(denominator + left);
            for (unsigned right = 0; right < choices; ++right)
            {
                const unsigned product = (denominator + left) * (denominator + right) - first_product;
                pairs[(heads[product] >> 4) + used[product]++] = uint16_t(left | (right << 8));
            }
        }
    }
};

struct RowAnchors
{
    unsigned diagonal = missing;
    unsigned off[4] = {missing, missing, missing, missing};
};

#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#define GLU_SCALE_HD __host__ __device__
#else
#define GLU_SCALE_HD
#endif

inline GLU_SCALE_HD REAL coefficient(unsigned grid_index)
{
    return REAL(denominator + grid_index) * (REAL(1) / REAL(denominator));
}

inline GLU_SCALE_HD unsigned product_index(REAL value, REAL base)
{
    const REAL approximate = value * (REAL(first_product) / base);
    if (!(approximate >= REAL(first_product) - REAL(0.5) && approximate < REAL(last_product) + REAL(0.5)))
        return missing;
    return static_cast<unsigned>(approximate + REAL(0.5)) - first_product;
}

inline GLU_SCALE_HD bool recover_row(const RowAnchors& anchors,
    const uint32_t* table,
    const uint16_t* pairs,
    const REAL* input,
    const REAL* base,
    uint16_t& coefficients)
{
    if (anchors.diagonal == missing)
        return false;
    const unsigned p = product_index(input[anchors.diagonal], base[anchors.diagonal]);
    if (p == missing)
        return false;
    const unsigned head = table[p], count = head & 15;
    const auto* diagonals = pairs + (head >> 4);
    unsigned survivors = (1u << count) - 1;
    for (unsigned a = 0; a < 4 && survivors && (survivors & (survivors - 1)); ++a)
    {
        const unsigned slot = anchors.off[a];
        if (slot == missing)
            break;
        const unsigned off_p = product_index(input[slot], base[slot]);
        if (off_p == missing)
            return false;
        const unsigned off_head = table[off_p], off_count = off_head & 15;
        const auto* off_pairs = pairs + (off_head >> 4);
        for (unsigned j = 0; j < count; ++j)
        {
            if (!(survivors & (1u << j)))
                continue;
            const unsigned row = diagonals[j] & 255;
            bool found = false;
            for (unsigned k = 0; k < off_count; ++k)
                found |= (off_pairs[k] & 255) == row;
            if (!found)
                survivors &= ~(1u << j);
        }
    }
    if (!survivors || (survivors & (survivors - 1)))
        return false;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    coefficients = diagonals[__ffs(int(survivors)) - 1];
#else
    coefficients = diagonals[__builtin_ctz(survivors)];
#endif
    return true;
}

inline GLU_SCALE_HD REAL rebuild_input(REAL base, FactorCoordinate position, const uint16_t* scales)
{
    const auto row = scales[position.row] & 255;
    const auto column = scales[position.column] >> 8;
    return (base * coefficient(row)) * coefficient(column);
}

inline GLU_SCALE_HD REAL rebuild_factor(REAL base,
    FactorCoordinate position,
    const uint16_t* scales,
    const REAL* reciprocals)
{
    const auto row = scales[position.row] & 255;
    const auto column = scales[position.column];
    const REAL multiplier = position.row > position.column ? reciprocals[column & 255] : coefficient(column >> 8);
    return (base * coefficient(row)) * multiplier;
}

#undef GLU_SCALE_HD

struct HostPlan
{
    std::vector<RowAnchors> rows;
    std::vector<FactorCoordinate> factor_coordinates;
    std::vector<REAL> base, expanded;
};

// A positive column weight leaves the rank of every leading block unchanged.
// For exact row/column scale factors R,C, undoing R,C on the rounded input
// perturbs each nonzero base entry by at most two normal multiply roundings.
// A 1% margin covers those errors and the double sums of nonnegative terms.
inline bool has_dominance_certificate(const HostPlan& plan, const std::vector<unsigned>& positions)
{
    const size_t n = plan.rows.size();
    std::vector<double> weights(n, 1.0), off_diagonal(n);
    // For H = |D|^-1 |offdiag(B)|, try positive weights (w + H*w)/2.
    // All row sums use the same old vector. Only the full inequality below
    // certifies a template. Four updates bound preparation on rejected inputs.
    constexpr unsigned max_weight_updates = 4;
    for (unsigned attempt = 0; attempt <= max_weight_updates; ++attempt)
    {
        std::fill(off_diagonal.begin(), off_diagonal.end(), 0.0);
        for (size_t p = 0; p < positions.size(); ++p)
        {
            const auto position = plan.factor_coordinates[positions[p]];
            if (position.row != position.column)
                off_diagonal[position.row] += std::abs(double(plan.base[p])) * weights[position.column];
        }
        bool dominant = true;
        for (size_t row = 0; row < n; ++row)
        {
            const double diagonal = std::abs(double(plan.base[plan.rows[row].diagonal]));
            if (!(diagonal * weights[row] > 1.01 * off_diagonal[row]))
                dominant = false;
            if (attempt < max_weight_updates)
                weights[row] = 0.5 * (weights[row] + off_diagonal[row] / diagonal);
        }
        if (dominant)
            return true;
    }
    return false;
}

inline HostPlan make_plan(const Symbolic_Matrix& a, const std::vector<unsigned>& positions)
{
    if (positions.empty() || !a.n || a.n > max_order)
        return {};
    HostPlan plan;
    plan.rows.resize(a.n);
    plan.factor_coordinates.resize(a.nnz);
    plan.base.resize(positions.size());
    plan.expanded.resize(a.nnz);
    std::vector<unsigned> compact_slot(a.nnz, missing);
    for (size_t p = 0; p < positions.size(); ++p)
    {
        const auto slot = positions[p];
        if (slot >= a.nnz || compact_slot[slot] != missing)
            throw std::invalid_argument("scale plan requires distinct compact positions");
        compact_slot[slot] = unsigned(p);
    }
    std::vector<unsigned> anchor_count(a.n);
    for (unsigned col = 0; col < a.n; ++col)
        for (unsigned slot = a.sym_c_ptr[col]; slot < a.sym_c_ptr[col + 1]; ++slot)
        {
            const unsigned row = a.sym_r_idx[slot];
            plan.factor_coordinates[slot] = {uint16_t(row), uint16_t(col)};
            const unsigned p = compact_slot[slot];
            if (p == missing)
                continue;
            const REAL base = a.val[slot];
            if (!std::isfinite(base) ||
                (base != REAL(0) && (std::abs(base) < REAL(1e-8) || std::abs(base) > REAL(1e8))))
                return {};
            plan.base[p] = base;
            plan.expanded[slot] = base;
            if (base == REAL(0))
                continue;
            if (slot == a.l_col_ptr[col])
                plan.rows[row].diagonal = p;
            else if (anchor_count[row] < 4)
                plan.rows[row].off[anchor_count[row]++] = p;
        }
    for (const auto& row : plan.rows)
        if (row.diagonal == missing)
            return {};
    if (!has_dominance_certificate(plan, positions))
        return {};
    return plan;
}

// Prequalify the base factorization before attempting any new-value reuse.
// A small residual and a bounded |L||U| norm give ample room for the two
// rounded scale products per factor and per input slot. Large-growth factors
// and extreme base values retain the original numerical path.
inline bool safe_base(const Symbolic_Matrix& a, const std::vector<REAL>& expanded, const std::vector<REAL>& factors)
{
    double greatest = 0;
    std::vector<double> matrix_rows(a.n), upper_rows(a.n);
    for (unsigned c = 0; c < a.n; ++c)
    {
        const auto diagonal = factors[a.l_col_ptr[c]];
        if (diagonal == REAL(0) || !std::isfinite(diagonal))
            return false;
        for (unsigned p = a.sym_c_ptr[c]; p < a.sym_c_ptr[c + 1]; ++p)
        {
            const REAL input = expanded[p], factor = factors[p];
            if (!std::isfinite(factor) || std::abs(factor) > std::numeric_limits<REAL>::max() / REAL(2))
                return false;
            const unsigned row = a.sym_r_idx[p];
            matrix_rows[row] += std::abs(double(input));
            if (p <= a.l_col_ptr[c])
                upper_rows[row] += std::abs(double(factor));
        }
    }
    auto product_rows = upper_rows;
    for (unsigned c = 0; c < a.n; ++c)
    {
        greatest = std::max(greatest, matrix_rows[c]);
        for (unsigned p = a.l_col_ptr[c] + 1; p < a.sym_c_ptr[c + 1]; ++p)
            product_rows[a.sym_r_idx[p]] += std::abs(double(factors[p])) * upper_rows[c];
    }
    for (const double row : product_rows)
        if (!std::isfinite(row) || row > 32 * greatest)
            return false;
    const double threshold = sizeof(REAL) == 8 ? 1e-12 : 1e-5;
    const double residual = a.factor_residual(factors.data(), expanded.data());
    return std::isfinite(residual) && residual <= threshold;
}

} // namespace diagonal_scale
