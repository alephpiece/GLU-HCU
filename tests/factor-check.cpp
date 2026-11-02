#include "symbolic.h"
#include <cmath>
#include <sstream>
#include <stdexcept>

void expect(double actual, double expected) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>1e-12)
        throw std::runtime_error("incorrect full factor residual");
}
int main() {
    try {
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
