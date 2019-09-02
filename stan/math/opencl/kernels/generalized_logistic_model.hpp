#ifndef STAN_MATH_OPENCL_KERNELS_GENERALIZED_LOGISTIC_MODEL_HPP
#define STAN_MATH_OPENCL_KERNELS_GENERALIZED_LOGISTIC_MODEL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_cl.hpp>
#include <stan/math/opencl/buffer_types.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {
// \cond
static const char *generalized_logistic_model_kernel_code = STRINGIFY(
    // \endcond
    /**
     * Matrix subtraction on the OpenCL device
     * Subtracts the second matrix from the
     * first matrix and stores the result
     * in the third matrix (C=A-B).
     *
     * @param[out] C The output matrix.
     * @param[in] B RHS input matrix.
     * @param[in] A LHS input matrix.
     * @param rows The number of rows for matrix A.
     * @param cols The number of columns for matrix A.
     * @param view_A triangular part of matrix A to use
     * @param view_B triangular part of matrix B to use
     * @note Code is a <code>const char*</code> held in
     * <code>subtract_kernel_code.</code>
     * Used in math/opencl/subtract_opencl.hpp
     *  This kernel uses the helper macros available in helpers.cl.
     */
    
    __kernel void generalized_logistic_model(__global double *tmp, __global double *IDp,
                           __global double *IDs, __global double *eta_ps, __global double *eta_ss,
                           __global double *eta_pr, __global double *eta_sr,
                           __global double *outtmp, __global double *outtmp1) {
      int i = get_global_id(0);
      int idp = IDp[i];
      int ids = IDs[i];
      outtmp[i] = tmp[8] + eta_ps[idp-1] + eta_ss[ids-1];
      outtmp1[i] = tmp[9] + eta_pr[idp - 1] + eta_sr[ids - 1];
      
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/subtract.hpp subtract() \endlink
 */
const kernel_cl<in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, out_buffer, out_buffer>
    generalized_logistic_model("generalized_logistic_model",
             {indexing_helpers, generalized_logistic_model_kernel_code});

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
