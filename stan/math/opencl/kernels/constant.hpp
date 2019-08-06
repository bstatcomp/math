#ifndef STAN_MATH_OPENCL_KERNELS_CONSTANT_HPP
#define STAN_MATH_OPENCL_KERNELS_CONSTANT_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_cl.hpp>
#include <stan/math/opencl/buffer_types.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {
// \cond
static const char* constant_kernel_code = STRINGIFY(
    // \endcond
    /**
     * Stores constant in the matrix on the OpenCL device.
     * Supports writing constants to the lower and upper triangular or
     * the whole matrix.
     *
     * @param[out] A matrix
     * @param rows Number of rows for matrix A
     * @param cols Number of columns for matrix A
     * @param part optional parameter that describes where to assign constant:
     *  LOWER - lower triangular
     *  UPPER - upper triangular
     * if the part parameter is not specified,
     * values are assigned to the whole matrix.
     * @note Code is a <code>const char*</code> held in
     * <code>constant_kernel_code.</code>
     * This kernel uses the helper macros available in helpers.cl.
     */
    __kernel void constants(__global double* A, double val, unsigned int rows,
                        unsigned int cols, unsigned int part) {
      int i = get_global_id(0);
      int j = get_global_id(1);
      if (i < rows && j < cols) {
        if ((part == LOWER && j < i) || (part == UPPER && j > i)
            || (part == ENTIRE)) {
          A(i, j) = val;
        }
      }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/constant.hpp constant() \endlink
 */
const kernel_cl<out_buffer, double, int, int, matrix_cl_view> constants(
    "constants", {indexing_helpers, constant_kernel_code});

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
