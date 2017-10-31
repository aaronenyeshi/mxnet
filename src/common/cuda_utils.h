/*!
 * Copyright (c) 2015 by Contributors
 * \file cuda_utils.h
 * \brief CUDA debugging utilities.
 */
#ifndef MXNET_COMMON_CUDA_UTILS_H_
#define MXNET_COMMON_CUDA_UTILS_H_


#include <dmlc/logging.h>
#include <mshadow/base.h>

#if MXNET_USE_CUDA
#include <hip-wrappers.h> // dummy include file placed in /opt/rocm/include
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include <hiprng.h>

namespace mxnet {
namespace common {
/*! \brief common utils for cuda */
namespace cuda {
/*!
 * \brief Get string representation of hipBLAS errors.
 * \param error The error.
 * \return String representation.
 */
inline const char* rocblasGetErrorString(rocblas_status error) {
  switch (error) {
  case rocblas_status_success:
    return "rocblas_status_success";
  case rocblas_status_invalid_handle :
    return "rocblas_status_invalid_handle";
  case rocblas_status_not_implemented:
    return "rocblas_status_not_implemented";
  case rocblas_status_invalid_pointer :
    return "rocblas_status_invalid_pointer";
  case rocblas_status_invalid_size:
    return "rocblas_status_invalid_size";
  case rocblas_status_memory_error :
    return "rocblas_status_memory_error";
  case rocblas_status_internal_error:
    return "rocblas_status_internal_error";
  default:
    break;
  }
  return "Unknown rocBLAS status";
}

/*!
 * \brief Get string representation of hipRNG errors.
 * \param status The status.
 * \return String representation.
 */
inline const char* HiprngGetErrorString(hiprngStatus_t status) {
  switch (status) {
  case HIPRNG_STATUS_SUCCESS:
    return "HIPRNG_STATUS_SUCCESS";
  case HIPRNG_STATUS_VERSION_MISMATCH:
    return "HIPRNG_STATUS_VERSION_MISMATCH";
  case HIPRNG_STATUS_NOT_INITIALIZED:
    return "HIPRNG_STATUS_NOT_INITIALIZED";
  case HIPRNG_STATUS_ALLOCATION_FAILED:
    return "HIPRNG_STATUS_ALLOCATION_FAILED";
  case HIPRNG_STATUS_TYPE_ERROR:
    return "HIPRNG_STATUS_TYPE_ERROR";
  case HIPRNG_STATUS_OUT_OF_RANGE:
    return "HIPRNG_STATUS_OUT_OF_RANGE";
  case HIPRNG_STATUS_LENGTH_NOT_MULTIPLE:
    return "HIPRNG_STATUS_LENGTH_NOT_MULTIPLE";
//  case HIPRNG_STATUS_DOUBLE_PRECISION_REQUIRED: // NOT SUPPORTED YET
//    return "HIPRNG_STATUS_DOUBLE_PRECISION_REQUIRED";
  case HIPRNG_STATUS_LAUNCH_FAILURE:
    return "HIPRNG_STATUS_LAUNCH_FAILURE";
  case HIPRNG_STATUS_PREEXISTING_FAILURE:
    return "HIPRNG_STATUS_PREEXISTING_FAILURE";
  case HIPRNG_STATUS_INITIALIZATION_FAILED:
    return "HIPRNG_STATUS_INITIALIZATION_FAILED";
  case HIPRNG_STATUS_ARCH_MISMATCH:
    return "HIPRNG_STATUS_ARCH_MISMATCH";
  case HIPRNG_STATUS_INTERNAL_ERROR:
    return "HIPRNG_STATUS_INTERNAL_ERROR";
  }
  return "Unknown hipRNG status";
}

}  // namespace cuda
}  // namespace common
}  // namespace mxnet

/*!
 * \brief Check CUDA error.
 * \param msg Message to print if an error occured.
 */
#define CHECK_CUDA_ERROR(msg)                                                \
  {                                                                          \
    hipError_t e = hipGetLastError();                                      \
    CHECK_EQ(e, hipSuccess) << (msg) << " CUDA: " << hipGetErrorString(e); \
  }

/*!
 * \brief Protected CUDA call.
 * \param func Expression to call.
 *
 * It checks for CUDA errors after invocation of the expression.
 */
#define CUDA_CALL(func)                                            \
  {                                                                \
    hipError_t e = (func);                                        \
    CHECK(e == hipSuccess)       \
        << "CUDA: " << hipGetErrorString(e);                      \
  }

/*!
 * \brief Protected hipBLAS call.
 * \param func Expression to call.
 *
 * It checks for hipBLAS errors after invocation of the expression.
 */
#define HIPBLAS_CALL(func)                                       \
  {                                                             \
    rocblas_status e = (func);                                  \
    CHECK_EQ(e, rocblas_status_success)                          \
        << "rocBLAS: " << common::cuda::rocblasGetErrorString(e); \
  }

/*!
 * \brief Protected hipRNG call.
 * \param func Expression to call.
 *
 * It checks for hipRNG errors after invocation of the expression.
 */
#define HIPRNG_CALL(func)                                       \
  {                                                             \
    hiprngStatus_t e = (func);                                  \
    CHECK_EQ(e, HIPRNG_STATUS_SUCCESS)                          \
        << "hipRNG: " << common::cuda::HiprngGetErrorString(e); \
  }

#endif  // MXNET_USE_CUDA

#if MXNET_USE_CUDNN

#include <miopen/miopen.h>

#define CUDNN_CALL(func)                                                      \
  {                                                                           \
    miopenStatus_t  e = (func);                                                 \
    CHECK_EQ(e, miopenStatusSuccess) << "miopen error code: " << e; \
  }

#endif  // MXNET_USE_CUDNN

// Overload atomicAdd to work for floats on all architectures
//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
#if (__HIP_DEVICE_COMPILE__) && (__HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__)
static inline __device__  void atomicAdd(double *address, double val) {
  unsigned long long* address_as_ull =                  // NOLINT(*)
    reinterpret_cast<unsigned long long*>(address);     // NOLINT(*)
  unsigned long long old = *address_as_ull;             // NOLINT(*)
  unsigned long long assumed;                           // NOLINT(*)

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                    __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
}
#endif

// Overload atomicAdd for half precision
// Taken from:
// https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomics.cuh
//#if defined(__CUDA_ARCH__)
#if (__HIP_DEVICE_COMPILE__)
static inline __device__ void atomicAdd(mshadow::half::half_t *address,
                                        mshadow::half::half_t val) {
  unsigned int *address_as_ui =
      reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) -
                                   (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    mshadow::half::half_t hsum;
    hsum.half_ =
        reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
    hsum += val;
    old = reinterpret_cast<size_t>(address) & 2
              ? (old & 0xffff) | (hsum.half_ << 16)
              : (old & 0xffff0000) | hsum.half_;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

#endif  // MXNET_COMMON_CUDA_UTILS_H_
