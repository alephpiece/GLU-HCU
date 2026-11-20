#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <numeric>
#include "Timer.h"
#include "symbolic.h"
#include <cmath>

using namespace std;

void Symbolic_Matrix::fill_in(unsigned *ai, unsigned *ap)
{
    // A reachable row above the diagonal contributes its already computed L
    // column. Visit each such column once, then sort the resulting row indices.
    vector<unsigned> multiplicity(n, 0);
    vector<unsigned> reached;
    sym_c_ptr.push_back(0);
    for (unsigned col = 0; col < n; ++col)
    {
        reached.clear();
        for (unsigned p = ap[col]; p < ap[col + 1]; ++p)
        {
            const unsigned row = ai[p];
            if (row >= n)
                throw invalid_argument("input row out of range");
            if (multiplicity[row]++ == 0)
                reached.push_back(row);
        }
        if (multiplicity[col] == 0)
        {
            multiplicity[col] = 1;
            reached.push_back(col);
        }

        for (size_t i = 0; i < reached.size(); ++i)
        {
            const unsigned source = reached[i];
            if (source >= col)
                continue;
            unsigned p = l_col_ptr[source];
            const unsigned end = sym_c_ptr[source + 1];
            while (p < end)
            {
                const unsigned first = p, row = sym_r_idx[p++];
                // Set union retains the larger multiplicity of a repeated row.
                while (p < end && sym_r_idx[p] == row)
                    ++p;
                if (multiplicity[row] == 0)
                    reached.push_back(row);
                multiplicity[row] = max(multiplicity[row], p - first);
            }
        }

        sort(reached.begin(), reached.end());
        for (const unsigned row : reached)
        {
            if (multiplicity[row] > numeric_limits<unsigned>::max() - sym_r_idx.size())
                throw overflow_error("symbolic pattern exceeds index range");
            if (row == col)
                l_col_ptr.push_back(sym_r_idx.size());
            sym_r_idx.insert(sym_r_idx.end(), multiplicity[row], row);
            multiplicity[row] = 0;
        }
        sym_c_ptr.push_back(sym_r_idx.size());
    }

    nnz = sym_c_ptr.back();
    m_out << "Symbolic nonzero: " << nnz << endl;
}

void Symbolic_Matrix::csr()
{
    csr_r_ptr.assign(size_t(n) + 1, 0);
    for (const unsigned row : sym_r_idx)
        ++csr_r_ptr[row + 1];
    partial_sum(csr_r_ptr.begin(), csr_r_ptr.end(), csr_r_ptr.begin());
    vector<unsigned> next(csr_r_ptr.begin(), csr_r_ptr.end() - 1);
    csr_c_idx.resize(csr_r_ptr.back());
    for (unsigned col = 0; col < n; ++col)
        for (unsigned p = sym_c_ptr[col]; p < sym_c_ptr[col + 1]; ++p)
        {
            const unsigned row = sym_r_idx[p], slot = next[row]++;
            csr_c_idx[slot] = col;
            if (row == col)
                csr_diag_ptr.push_back(slot);
        }
}

// Scatter one input column into a row-indexed buffer, then gather its filled
// column. In reverse order the first supplied value wins for duplicate rows.
void Symbolic_Matrix::predictLU(unsigned *ai, unsigned *ap, double *ax)
{
    vector<REAL> values(n, REAL(0));
    val.reserve(nnz);
    for (unsigned col = 0; col < n; ++col)
    {
        for (unsigned p = ap[col + 1]; p > ap[col]; --p)
        {
            const unsigned row = ai[p - 1];
            if (row >= n)
                throw invalid_argument("input row out of range");
            values[row] = static_cast<REAL>(ax[p - 1]);
        }
        for (unsigned p = sym_c_ptr[col]; p < sym_c_ptr[col + 1]; ++p)
            val.push_back(values[sym_r_idx[p]]);
        for (unsigned p = ap[col]; p < ap[col + 1]; ++p)
            values[ai[p]] = REAL(0);
    }
}

void Symbolic_Matrix :: leveling()
{
    vector<int> inlevel, level_size(n, 0);
    inlevel.reserve(n);

    for (unsigned i = 0; i < n; ++i)
    {
        int max_lv = -1, lv;
        //search dependent columns on the left
        for (unsigned j = sym_c_ptr[i]; j < l_col_ptr[i]; ++j) {
            unsigned nz_idx = sym_r_idx[j]; //Nonzero row in col i, U part

            //L part of col nz_idx exists , U-dependency found
            if (l_col_ptr[nz_idx] + 1 != sym_c_ptr[nz_idx+1]) {
                lv = inlevel[nz_idx];
                if (lv > max_lv)
                    max_lv = lv;
            }
        }
        for (unsigned j = csr_r_ptr[i]; j < csr_diag_ptr[i]; ++j) {
            unsigned nz_idx = csr_c_idx[j];
            lv = inlevel[nz_idx];
            if (lv > max_lv)
                max_lv = lv;
        }        
        lv = max_lv + 1;
        inlevel.push_back(lv);
        ++level_size[lv];
        if (lv > num_lev)
            num_lev = lv;
    }

    ++num_lev;

    level_ptr.reserve(num_lev);
    level_ptr.push_back(0);
    for (int i = 0; i < num_lev; ++i)
        level_ptr.push_back(level_ptr[i] + level_size[i]);

    level_idx.resize(n);
    vector<int> tlen(level_ptr);
    for (unsigned i = 0; i < n; ++i)
        level_idx[tlen[inlevel[i]]++] = i;

    m_out << "Number of levels: " << num_lev << endl;
}

#if GLU_DEBUG
void Symbolic_Matrix::PrintLevel()
{
    for (int i = 0; i < num_lev; ++i) {
        for (int j = level_ptr[i]; j < level_ptr[i + 1]; ++j)
            m_out << level_idx[j] << ' ';
        m_out << '\n';
    }
}

void Symbolic_Matrix::ABFTCalculateCCA()
{
    CCA.assign(n, 0.0);
    for (unsigned i = 0; i < n; ++i) {
        for (unsigned j = sym_c_ptr[i]; j < sym_c_ptr[i + 1]; ++j)
            CCA[i] += val[j];
    }
}

void Symbolic_Matrix::ABFTCheckResult()
{
     vector<REAL> CCL(n, 1.0);
     for(unsigned i = 0; i < n; i++)
        for(unsigned j = l_col_ptr[i] + 1; j < sym_c_ptr[i + 1]; j++)
            CCL[i] += val[j];

     vector<REAL> CCA_ABFT(n, 0.0);
     for(unsigned i = 0; i < n; i++)
     {
        CCA_ABFT[i] = 0;
        for(unsigned j = sym_c_ptr[i]; j <= l_col_ptr[i]; j++)
            CCA_ABFT[i] += CCL[sym_r_idx[j]] * val[j];
     }

     //Compare CCA and CCA_ABFT
     for(unsigned i = 0; i < n; ++i) {
         if (abs(CCA_ABFT[i] - CCA[i]) > 1e-5) {
             m_err << "Column " << i << ": CCA = " << CCA[i] <<
                 ", CCA_ABFT[i] = " << CCA_ABFT[i] << endl;
             m_err << "More to come..." << endl;
             return;
         }
     }
     m_out << "Results passed ABFT check." << endl;
}
#endif

vector<REAL> Symbolic_Matrix::solve(SNicsLU *nicslu, const vector<REAL> &rhs)
{
    return solve(nicslu, rhs, val.data());
}

vector<REAL> Symbolic_Matrix::solve(SNicsLU *nicslu, const vector<REAL> &rhs, const REAL* factors) const
{
    vector<REAL> b(n);
    vector<REAL> x(n);
    if (n == 0)
        return x;

    unsigned mc64_scale = nicslu->cfgi[1];

    unsigned *rp = nicslu->col_perm;
    unsigned *cp = nicslu->row_perm_inv;
    int *piv = nicslu->pivot;
    double *rows = nicslu->col_scale_perm;
    double *cols = nicslu->row_scale;

    //apply row permutation and row scaling to rhs
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j) {
            unsigned p = piv[j];
            b[j] = rhs[rp[piv[j]]] * rows[p];
        }
    else
        for (unsigned j = 0; j < n; ++j)
            b[j] = rhs[rp[piv[j]]];

    //left-multiply inv(L)
    for (unsigned j = 0; j < n; ++j) {
        for (unsigned p = l_col_ptr[j] + 1; p < sym_c_ptr[j + 1]; ++p)
            b[sym_r_idx[p]] -= factors[p] * b[j];
    }

    //left-multiply inv(U)
    for (int jj = n - 1; jj >= 0; --jj) {
        unsigned diag = l_col_ptr[jj];
        b[jj] /= factors[diag];

        for (unsigned p = sym_c_ptr[jj]; p < l_col_ptr[jj]; ++p) {
            b[sym_r_idx[p]] -= factors[p] * b[jj];
        }
    }

    //apply col permutation and scaling
    if (mc64_scale)
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]] * cols[j];
    else
        for (unsigned j = 0; j < n; ++j)
            x[j] = b[cp[j]];


    return x;
}

// Full ||B-LU||_inf / ||B||_inf, for the preprocessed FP matrix B.
// Accumulate one sparse product column in long double; include product entries
// outside B's pattern. No dense n*n matrix, sampling, or checksum substitution.
double Symbolic_Matrix::factor_residual(const REAL* lu) const {
    const auto& a = *this;
    std::vector<long double> column(a.n,0), residual_rows(a.n,0), matrix_rows(a.n,0);
    std::vector<unsigned> touched;
    std::vector<int> mark(a.n,-1);
    auto add = [&](unsigned row, unsigned col, long double value) {
        if (mark[row] != int(col)) { mark[row]=int(col); touched.push_back(row); }
        column[row] += value;
    };
    for (unsigned j=0; j<a.n; ++j) {
        touched.clear();
        for (unsigned p=a.sym_c_ptr[j]; p<=a.l_col_ptr[j]; ++p) {
            const unsigned k=a.sym_r_idx[p];
            const long double u=lu[p];
            add(k,j,u); // implicit unit diagonal in L
            for (unsigned q=a.l_col_ptr[k]+1; q<a.sym_c_ptr[k+1]; ++q)
                add(a.sym_r_idx[q],j,static_cast<long double>(lu[q])*u);
        }
        for (unsigned p=a.sym_c_ptr[j]; p<a.sym_c_ptr[j+1]; ++p) {
            add(a.sym_r_idx[p],j,-static_cast<long double>(a.val[p]));
            matrix_rows[a.sym_r_idx[p]] += std::abs(static_cast<long double>(a.val[p]));
        }
        for (unsigned i : touched) {
            if (!std::isfinite(column[i])) throw std::runtime_error("nonfinite LU product");
            residual_rows[i] += std::abs(column[i]);
            column[i]=0;
        }
    }
    const long double denominator=*std::max_element(matrix_rows.begin(),matrix_rows.end());
    return double(*std::max_element(residual_rows.begin(),residual_rows.end())/denominator);
}
