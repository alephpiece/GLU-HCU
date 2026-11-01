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
add_executable(lu_cmd src/lu_cmd.cpp src/numeric.cu src/symbolic.cc
    src/Timer.cpp src/preprocess/preprocess.c)
glu_configure_target(lu_cmd)

message(STATUS "GLU backend: ${GLU_BACKEND}; FP32: ${GLU_USE_FP32}")
