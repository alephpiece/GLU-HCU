#include "symbolic.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

void expect(double actual, double expected) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>1e-12)
        throw std::runtime_error("incorrect full factor residual");
}
// Unsorted and repeated rows exercise both structural union and the mapping
// from the original entries to the filled columns, including signed zero.
void check_symbolic_preparation() {
    std::ostringstream out, err;
    Symbolic_Matrix a(5, out, err);
    unsigned rows[]{2,1,0,2, 3,1, 0, 4,3};
    unsigned columns[]{0,4,6,7,9,9};
    double values[]{-0.0,2,3,8, 4,5, 6, 7,9};
    a.fill_in(rows, columns);
    a.csr();
    a.predictLU(rows, columns, values);
    if (a.sym_c_ptr != std::vector<unsigned>{0,4,6,11,13,14} ||
        a.sym_r_idx != std::vector<unsigned>{0,1,2,2, 1,3, 0,1,2,2,3, 3,4, 4} ||
        a.l_col_ptr != std::vector<unsigned>{0,4,8,11,13} ||
        a.csr_r_ptr != std::vector<unsigned>{0,2,5,9,12,14} ||
        a.csr_c_idx != std::vector<unsigned>{0,2, 0,1,2, 0,0,2,2, 1,2,3, 3,4} ||
        a.csr_diag_ptr != std::vector<unsigned>{0,3,7,8,11,13})
        throw std::runtime_error("symbolic preparation changed the filled pattern");
    if (a.val != std::vector<REAL>{3,2,0,0, 5,4, 6,0,0,0,0, 9,7, 0} ||
        !std::signbit(a.val[2]) || !std::signbit(a.val[3]) || std::signbit(a.val[8]))
        throw std::runtime_error("symbolic preparation changed an input or fill value");

    unsigned bad_row[]{1}, valid_row[]{0}, single_column[]{0,1};
    double value[]{1};
    bool rejected = false;
    Symbolic_Matrix bad(1, out, err);
    try { bad.fill_in(bad_row, single_column); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("fill accepted an invalid input row");

    Symbolic_Matrix valid(1, out, err);
    valid.fill_in(valid_row, single_column);
    try { valid.predictLU(bad_row, single_column, value); }
    catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("value initialization accepted an invalid input row");
}

int main() {
    try {
        check_symbolic_preparation();
        std::ostringstream out, err;
        Symbolic_Matrix a(2,out,err);
        a.nnz=4;
        a.sym_c_ptr={0,2,4}; a.sym_r_idx={0,1,0,1}; a.l_col_ptr={0,3};
        a.val={4,1,2,3};
        REAL lu[]{4,0.25,2,2.5};
        expect(a.factor_residual(lu),0);
        lu[1]=0.5;
        expect(a.factor_residual(lu),0.25);
        // L21*U13 produces a nonzero at (2,3) outside this supplied pattern.
        Symbolic_Matrix b(3,out,err);
        b.nnz=5;
        b.sym_c_ptr={0,2,3,5}; b.sym_r_idx={0,1,1,0,2}; b.l_col_ptr={0,2,4};
        b.val={1,1,1,1,1};
        expect(b.factor_residual(b.val.data()),0.5);
        // A structural fill entry has zero original value but nonzero U.
        b.nnz=6;
        b.sym_c_ptr={0,2,3,6}; b.sym_r_idx={0,1,1,0,1,2}; b.l_col_ptr={0,2,5};
        b.val={1,1,1,1,0,1};
        REAL filled[]{1,1,1,1,-1,1};
        expect(b.factor_residual(filled),0);
        std::cout << "Full factor residual positive/negative/fill tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
