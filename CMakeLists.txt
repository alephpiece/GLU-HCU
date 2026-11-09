cmake_minimum_required(VERSION 3.21)
project(glu LANGUAGES C CXX)

set(GLU_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA or HIP")
set_property(CACHE GLU_BACKEND PROPERTY STRINGS CUDA HIP)
string(TOUPPER "${GLU_BACKEND}" GLU_BACKEND)
if(NOT GLU_BACKEND MATCHES "^(CUDA|HIP)$")
    message(FATAL_ERROR "GLU_BACKEND must be CUDA or HIP")
endif()
enable_language(${GLU_BACKEND})
option(GLU_USE_FP32 "Use FP32 instead of the default FP64" OFF)

find_package(Threads REQUIRED)
# NICSLU's existing Unix build compiles all these sources with these definitions.
file(GLOB NICSLU_SOURCES CONFIGURE_DEPENDS src/nicslu/source/*.c)
add_library(nicslu STATIC ${NICSLU_SOURCES} src/nicslu/util/nicslu_readfile.c)
target_include_directories(nicslu PUBLIC src/nicslu/include src/nicslu/util)
target_compile_definitions(nicslu PRIVATE NO_ATOMIC SSE2)
target_compile_options(nicslu PRIVATE -msse2)
target_link_libraries(nicslu PUBLIC Threads::Threads m)

function(glu_configure_target target)
    target_include_directories(${target} PRIVATE include src/preprocess)
    target_link_libraries(${target} PRIVATE nicslu)
    target_compile_definitions(${target} PRIVATE GLU_DEBUG=0)
    if(GLU_USE_FP32)
        target_compile_definitions(${target} PRIVATE GLU_USE_FP32=1)
    endif()
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
        HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
    if(GLU_BACKEND STREQUAL "HIP")
        target_compile_definitions(${target} PRIVATE GLU_USE_HIP=1)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:--gpu-max-threads-per-block=1024>)
    endif()
endfunction()

set_source_files_properties(src/numeric.cu PROPERTIES LANGUAGE ${GLU_BACKEND})
set(GLU_COMMON_SOURCES src/numeric.cu src/symbolic.cc src/Timer.cpp src/preprocess/preprocess.c)
add_executable(lu_cmd src/lu_cmd.cpp ${GLU_COMMON_SOURCES})
glu_configure_target(lu_cmd)
add_executable(glu_bench src/lu_bench.cpp ${GLU_COMMON_SOURCES})
glu_configure_target(glu_bench)

option(GLU_BUILD_TESTS "Build GPU numeric regression tests" OFF)
if(GLU_BUILD_TESTS)
    enable_testing()
    add_executable(factor-check-tests tests/factor-check.cpp src/symbolic.cc)
    glu_configure_target(factor-check-tests)
    add_test(NAME factor-check COMMAND factor-check-tests)
    set_source_files_properties(tests/gpu-kernels.cu PROPERTIES LANGUAGE ${GLU_BACKEND})
    add_executable(gpu-kernel-tests tests/gpu-kernels.cu src/symbolic.cc)
    glu_configure_target(gpu-kernel-tests)
    add_test(NAME gpu-kernels COMMAND gpu-kernel-tests)
    set_tests_properties(gpu-kernels PROPERTIES TIMEOUT 120)
    add_executable(fused-execution-tests tests/fused-execution.cpp
        src/numeric.cu src/symbolic.cc)
    glu_configure_target(fused-execution-tests)
    add_test(NAME fused-execution COMMAND fused-execution-tests)
    set_tests_properties(fused-execution PROPERTIES TIMEOUT 120)
    add_executable(task-set-tests tests/task-set.cpp src/numeric.cu src/symbolic.cc)
    glu_configure_target(task-set-tests)
    add_test(NAME task-set COMMAND task-set-tests)
    set_tests_properties(task-set PROPERTIES TIMEOUT 120)
    add_test(NAME cli-output COMMAND ${CMAKE_COMMAND}
        -DGLU_EXECUTABLE=$<TARGET_FILE:lu_cmd>
        -DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/cli-output-test
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli-output.cmake)
    set_tests_properties(cli-output PROPERTIES TIMEOUT 120)
    add_test(NAME benchmark COMMAND ${CMAKE_COMMAND}
        -DGLU_BENCH=$<TARGET_FILE:glu_bench>
        -DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/benchmark-test
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark.cmake)
    set_tests_properties(benchmark PROPERTIES TIMEOUT 120)
endif()
message(STATUS "GLU backend: ${GLU_BACKEND}; FP32: ${GLU_USE_FP32}")
