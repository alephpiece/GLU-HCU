#pragma once

#if defined(GLU_USE_HIP)
#include <hip/hip_runtime.h>
using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaStream_t = hipStream_t;
using cudaGraph_t = hipGraph_t;
using cudaGraphExec_t = hipGraphExec_t;
using cudaDeviceProp = hipDeviceProp_t;
#define cudaSuccess hipSuccess
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaMallocHost hipHostMalloc
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#else
#include <cuda_runtime.h>
#endif

// Logical column-processing group, deliberately identical on both backends.
// A wave64 contains two independent groups. Every lane in a group must call
// these collectives, including lanes beyond the end of a sparse column.
constexpr unsigned GLU_GROUP_SIZE = 32;

__device__ inline unsigned gpuBallot32(bool predicate)
{
#if defined(GLU_USE_HIP)
    const unsigned shift = (threadIdx.x % warpSize) / GLU_GROUP_SIZE * GLU_GROUP_SIZE;
    return static_cast<unsigned>(__ballot(predicate) >> shift);
#else
    return __ballot_sync(0xffffffffu, predicate);
#endif
}

template <typename T>
__device__ inline T gpuShuffle32(T value, int source_lane)
{
#if defined(GLU_USE_HIP)
    return __shfl(value, source_lane, GLU_GROUP_SIZE);
#else
    return __shfl_sync(0xffffffffu, value, source_lane, GLU_GROUP_SIZE);
#endif
}
