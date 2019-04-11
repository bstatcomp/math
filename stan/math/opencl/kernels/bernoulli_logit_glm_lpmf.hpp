
#ifndef STAN_MATH_OPENCL_KERNELS_BERNOULLI_LOGIT_GLM_LPMF_HPP
#define STAN_MATH_OPENCL_KERNELS_BERNOULLI_LOGIT_GLM_LPMF_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_cl.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {

// \cond
static const char *bernoulli_logit_glm_kernel_code = STRINGIFY(
// \endcond
        /**
         * GPU implementation of Generalized Linear Model (GLM)
         * with Bernoulli distribution and logit link function.
         *
         * Must be run with at least N threads and local size equal to LOCAL_SIZE_.
         * @param[in] y_glob binary vector parameter
         * @param[in] x design matrix
         * @param[in] alpha intercept (in log odds)
         * @param[in] beta weight vector
         * @param[out] logp_glob partially summed log probabiltiy (1 value per work group)
         * @param[out] theta_derivative_glob intermediate variable used in the model
         * @param[out] theta_derivative_sum partially summed theta_derivative_glob (1 value per work group)
         * @param N number of cases
         * @param M number of attributes
         * @param is_alpha_vector 0 or 1 - whether alpha is a vector (alternatively it is a scalar)
         * @param need_theta_derivative interpreted as boolean - whether theta_derivative needs to be computed
         * @param need_theta_derivative_sum interpreted as boolean - whether theta_derivative_sum needs to be computeds
         */
        __kernel void bernoulli_logit_glm(const __global double* y_glob, const __global double* x, const __global double* alpha, const __global double* beta,
                                          __global double* logp_glob, __global double* theta_derivative_glob, __global double* theta_derivative_sum,
                                          const int N, const int M, const int is_alpha_vector, const int need_theta_derivative, const int need_theta_derivative_sum) {
          const int gid = get_global_id(0);
          const int lid = get_local_id(0);
          const int lsize = get_local_size(0);
          const int wgid = get_group_id(0);

          __local double res_loc[LOCAL_SIZE_];

          double logp=0;
          double theta_derivative=0;
          if(gid<N){
            double ytheta=0;
            for (int i = 0, j = 0; i < M; i++, j += N) {
              ytheta += x[j + gid] * beta[i];
            }
            const double y = y_glob[gid];
            const double sign_ = 2 * y - 1;
            ytheta += alpha[gid*is_alpha_vector];
            ytheta *= sign_;
            if(y > 1 || y < 0 || !isfinite(ytheta)){
              logp=NAN;
            }
            const double exp_m_ytheta = exp(-ytheta);

            const double cutoff = 20.0;
            if(ytheta>cutoff){
              logp -= exp_m_ytheta;
              theta_derivative = -exp_m_ytheta;
            }
            else if (ytheta < -cutoff){
              logp += ytheta;
              theta_derivative = sign_;
            }
            else{
              logp += -log1p(exp_m_ytheta);
              theta_derivative = sign_ * exp_m_ytheta / (exp_m_ytheta+1);
            }

            if(need_theta_derivative){
              theta_derivative_glob[gid] = theta_derivative;
            }
            res_loc[lid] = logp;
          }
          else{
            res_loc[lid] = 0;
          }
          barrier(CLK_LOCAL_MEM_FENCE);
          for (int step = lsize / REDUCTION_STEP_SIZE; step > 0; step /= REDUCTION_STEP_SIZE) {
            if (lid < step) {
              for (int i = 1; i < REDUCTION_STEP_SIZE; i++) {
                res_loc[lid] += res_loc[lid + step * i];
              }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
          }
          if (lid == 0) {
            logp_glob[wgid] = res_loc[0];
          }


          if(need_theta_derivative_sum){
            barrier(CLK_LOCAL_MEM_FENCE);
            if(gid<N){
              res_loc[lid] = theta_derivative;
            }
            else{
              res_loc[lid] = 0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int step = lsize / REDUCTION_STEP_SIZE; step > 0; step /= REDUCTION_STEP_SIZE) {
              if (lid < step) {
                for (int i = 1; i < REDUCTION_STEP_SIZE; i++) {
                  res_loc[lid] += res_loc[lid + step * i];
                }
              }
              barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
              theta_derivative_sum[wgid] = res_loc[0];
            }
          }
        }
// \cond
);
// \endcond

/**
 * See the docs for \link kernels/bernoulli_logit_glm_lpmf.hpp bernoulli_logit_glm() \endlink
 */
const local_range_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int, int, int, int, int>
        bernoulli_logit_glm("bernoulli_logit_glm", bernoulli_logit_glm_kernel_code, {{"REDUCTION_STEP_SIZE",4},{"LOCAL_SIZE_", 64}});


}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan

#endif
#endif
