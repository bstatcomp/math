#ifndef STAN_MATH_OPENCL_KERNELS_DIAGONAL_MATRIX_HPP
#define STAN_MATH_OPENCL_KERNELS_DIAGONAL_MATRIX_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_cl.hpp>
#include <stan/math/opencl/buffer_types.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {
// \cond
static const char *diag_matrix_kernel_code = STRINGIFY(
// \endcond
/**

 */
        __kernel void diag_matrix(__global double *M, __global double *V, int rows) {
          int i = get_global_id(0);
          M[i + rows * i] = V[i];
        }
// \cond
);
// \endcond

/**
 * See the docs for \link kernels/diag_matrix.hpp diag_matrix() \endlink
 */
const kernel_cl<out_buffer, in_buffer, int, int>
        diag_matrix("diag_matrix",
                 {diag_matrix_kernel_code});

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
