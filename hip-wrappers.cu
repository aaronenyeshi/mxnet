/* Dummy header file to add types and fucntions to silence compiler errors.
   This needs to be replaced with working fixes in actual header files and         libraries.
*/

#include "hip/hip_runtime.h"
#include "hip-wrappers.h"



rocblas_status rocblas_sgemmEx  (rocblas_handle handle,
                                 rocblas_operation transa,
                                 rocblas_operation transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float *alpha, /* host or device pointer */  
                                 const void *A,
                                 hipDataType Atype,
                                 int lda,
                                 const void *B,
                                 hipDataType Btype,
                                 int ldb,
                                 const float *beta, /* host or device pointer */
                                 void *C,
                                 hipDataType Ctype,
                                 int ldc)
{
	return rocblas_status_not_implemented;
}
