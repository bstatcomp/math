#ifndef STAN_MATH_OPENCL_CONVERT_INT_TO__DOUBLE_HPP
#define STAN_MATH_OPENCL_CONVERT_INT_TO__DOUBLE_HPP

#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_cl.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {
// \cond
static const char *convert_int_to_double_kernel_code = STRINGIFY(
    // \endcond
    /**
     * Convert matrix of integers to a matrix of doubles.
     * @param[in] A The matrix to convert.
     * @param[out] B The matrix to place result into.
     */
    __kernel void convert_int_to_double(__global int *A, __global double *B) {
      int i = get_global_id(0);
      B[i] = A[i];
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/convert_int_to_double.hpp convert_int_to_double() \endlink
 */
const global_range_kernel<cl::Buffer, cl::Buffer> convert_int_to_double(
    "convert_int_to_double", convert_int_to_double_kernel_code);

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif

#endif
