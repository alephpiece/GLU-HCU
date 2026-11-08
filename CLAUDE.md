# GLU repository guidance

GLU combines CPU preprocessing and symbolic analysis with GPU sparse LU factorization. The CUDA and HIP backends share the numerical implementation. `lu_cmd` solves a single system, while `glu_bench` measures repeated factorization workloads.

## Documentation

Build configuration and regression test commands are maintained in [BUILDING.md](BUILDING.md). The [documentation index](docs/README.md) links the solver, benchmark and C++ interface manuals, together with the original algorithm publications.

## Source map

- [src/preprocess/](src/preprocess/): matrix input, permutation and scaling through the bundled NICSLU library.
- [src/symbolic.cc](src/symbolic.cc) and [include/symbolic.h](include/symbolic.h): fill pattern, dependency levels, factor residuals and CPU triangular solves.
- [src/numeric.cu](src/numeric.cu) and [include/numeric.h](include/numeric.h): GPU numerical execution, reusable matrix templates and asynchronous tasks.
- [src/lu_cmd.cpp](src/lu_cmd.cpp) and [src/lu_bench.cpp](src/lu_bench.cpp): solver and benchmark entry points.
- [tests/](tests/): numerical, task-lifecycle and command-line regressions.
- [src/nicslu/](src/nicslu/): bundled third-party library and its documentation.

## Maintenance

- Use the documented build options to select backend and precision, keeping CPU and GPU translation units consistent.
- When changing numerical execution, check dependencies, buffer ownership and asynchronous lifetimes, and run the affected regressions for the relevant backend and precision.
- Keep public interface contracts in `include/numeric.h`, command-line help and the corresponding manuals in `docs/` consistent with implementation changes.
