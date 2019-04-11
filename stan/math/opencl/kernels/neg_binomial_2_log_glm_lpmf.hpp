#ifndef STAN_MATH_OPENCL_KERNELS_NEG_BINOMIAL_2_LOG_GLM_LPMF_HPP
#define STAN_MATH_OPENCL_KERNELS_NEG_BINOMIAL_2_LOG_GLM_LPMF_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_cl.hpp>

namespace stan {
namespace math {
namespace opencl_kernels {

// \cond
static const char *neg_binomial_2_log_glm_kernel_code = STRINGIFY(
// \endcond
        /**
         * Caldulates the digamma function - derivative of logarithm of gamma. This implementation is based on one from boost 1.69.0.
         * @param x point at which to calculate digamma
         * @return digamma(x)
         */
        double digamma(double x) {
          double result = 0;
          if (x <= -1) {
            x = 1 - x;
            double remainder = x - floor(x);
            if (remainder > 0.5) {
              remainder -= 1;
            }
            if (remainder == 0) {
              return NAN;
            }
            result = M_PI / tan(M_PI * remainder);
          }
          if (x == 0) {
            return NAN;
          }
          //in boost: x >= digamma_large_lim(t)
          if (x > 10) {
            //in boost: result += digamma_imp_large(x, t);
            double P[8];
            P[0] = 0.083333333333333333333333333333333333333333333333333;
            P[1] = -0.0083333333333333333333333333333333333333333333333333;
            P[2] = 0.003968253968253968253968253968253968253968253968254;
            P[3] = -0.0041666666666666666666666666666666666666666666666667;
            P[4] = 0.0075757575757575757575757575757575757575757575757576;
            P[5] = -0.021092796092796092796092796092796092796092796092796;
            P[6] = 0.083333333333333333333333333333333333333333333333333;
            P[7] = -0.44325980392156862745098039215686274509803921568627;
            x -= 1;
            result += log(x);
            result += 1 / (2 * x);
            double z = 1 / (x * x);
            double tmp = P[7];
            for (int i = 6; i >= 0; i--) {
              tmp = tmp * z + P[i];
            }
            //tmp=boost::tools::evaluate_polynomial(P, z);
            result -= z * tmp;
          }
          else {
            while (x > 2) {
              x -= 1;
              result += 1 / x;
            }
            while (x < 1) {
              result -= 1 / x;
              x += 1;
            }
            //in boost: result += digamma_imp_1_2(x, t);
            const float Y = 0.99558162689208984F;

            const double root1 = (double)1569415565 / 1073741824uL;
            const double root2 = (double)381566830 / 1073741824uL / 1073741824uL;
            const double root3 = 0.9016312093258695918615325266959189453125e-19;

            double P[6];
            P[0] = 0.25479851061131551;
            P[1] = -0.32555031186804491;
            P[2] = -0.65031853770896507;
            P[3] = -0.28919126444774784;
            P[4] = -0.045251321448739056;
            P[5] = -0.0020713321167745952;
            double Q[7];
            Q[0] = 1.0;
            Q[1] = 2.0767117023730469;
            Q[2] = 1.4606242909763515;
            Q[3] = 0.43593529692665969;
            Q[4] = 0.054151797245674225;
            Q[5] = 0.0021284987017821144;
            Q[6] = -0.55789841321675513e-6;
            double g = x - root1 - root2 - root3;
            double tmp = P[5];
            for (int i = 4; i >= 0; i--) {
              tmp = tmp * (x - 1) + P[i];
            }
            double tmp2 = Q[6];
            for (int i = 5; i >= 0; i--) {
              tmp2 = tmp2 * (x - 1) + Q[i];
            }
            // in boost: T r = tools::evaluate_polynomial(P, T(x-1)) / tools::evaluate_polynomial(Q, T(x-1));
            double r = tmp / tmp2;
            result += g * Y + g * r;
          }
          return result;
        }
        /**
         * GPU implementation of Generalized Linear Model (GLM)
         * with Poisson distribution and log link function.
         *
         * Must be run with at least N threads and local size equal to LOCAL_SIZE_.
         * @param[in] y_glob positive integer vector parameter
         * @param[in] x design matrix
         * @param[in] alpha intercept (in log odds)
         * @param[in] beta weight vector
         * @param[out] theta_derivative_glob intermediate variable used in the model
         * @param[out] theta_derivative_sum partially summed theta_derivative_glob (1 value per work group)
         * @param[out] logp1_glob partially summed part of log probabiltiy (1 value per work group)
         * @param[out] logp2_glob partially summed part of log probabiltiy (1 value per work group)
         * @param N number of cases
         * @param M number of attributes
         * @param is_alpha_vector 0 or 1 - whether alpha is a vector (alternatively it is a scalar)
         * @param need_logp1 interpreted as boolean - whether logp1_glob needs to be computed
         * @param need_logp2 interpreted as boolean - whether logp2_glob needs to be computed
         */
        __kernel void neg_binomial_2_log_glm(const __global double* y_glob, const __global double* x, const __global double* alpha, const __global double* beta, const __global double* phi_glob,
                                             __global double* logp_glob, __global double* theta_derivative_glob, __global double* theta_derivative_sum, __global double* phi_derivative_glob,
                                      const int N, const int M, const int is_alpha_vector, const int is_phi_vector,
                                      const int need_theta_derivative, const int need_theta_derivative_sum, const int need_phi_derivative, const int need_phi_derivative_sum,
                                      const int need_logp1, const int need_logp2, const int need_logp3, const int need_logp4, const int need_logp5) {
          const int gid = get_global_id(0);
          const int lid = get_local_id(0);
          const int lsize = get_local_size(0);
          const int wgid = get_group_id(0);

          __local double res_loc[LOCAL_SIZE_];
          double logp=0;
          double phi_derivative=0;
          double theta_derivative=0;
          if(gid<N){
            double theta = 0;
            for (int i = 0, j = 0; i < M; i++, j += N) {
              theta += x[j + gid] * beta[i];
            }
            double phi = phi_glob[gid*is_phi_vector];
            double y = y_glob[gid];
            if(!isfinite(theta) || y < 0 || !isfinite(y) || !isfinite(phi)){
              logp=NAN;
            }
            theta += alpha[gid*is_alpha_vector];
            double log_phi = log(phi);
            double logsumexp_theta_logphi;
            if(theta > log_phi){
              logsumexp_theta_logphi = theta + log1p(exp(log_phi - theta));
            }
            else{
              logsumexp_theta_logphi = log_phi + log1p(exp(theta - log_phi));
            }
            double y_plus_phi = y+phi;
            if(need_logp1){
              logp -= lgamma(y+1);
            }
            if(need_logp2){
              logp -= lgamma(phi);
              if(phi!=0){
                logp += phi * log(phi);
              }
            }
            if(need_logp3){
              logp -= y_plus_phi * logsumexp_theta_logphi;
            }
            if(need_logp4){
              logp += y * theta;
            }
            if(need_logp5){
              logp += lgamma(y_plus_phi);
            }
            double theta_exp = exp(theta);
            theta_derivative = y - theta_exp * y_plus_phi / (theta_exp + phi);
            if(need_theta_derivative){
              theta_derivative_glob[gid]=theta_derivative;
            }
            if(need_phi_derivative){
              phi_derivative = 1 - y_plus_phi / (theta_exp + phi) + log_phi - logsumexp_theta_logphi + digamma(y_plus_phi) - digamma(phi);
              if(!need_phi_derivative_sum){
                phi_derivative_glob[gid]=phi_derivative;
              }
            }
          }

          if(need_logp1 || need_logp2 || need_logp3 || need_logp4 || need_logp5){
            barrier(CLK_LOCAL_MEM_FENCE);
            if(gid<N){
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

          if(need_phi_derivative_sum){
            barrier(CLK_LOCAL_MEM_FENCE);
            if(gid<N){
              res_loc[lid] = phi_derivative;
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
              phi_derivative_glob[wgid] = res_loc[0];
            }
          }
        }
// \cond
);
// \endcond

/**
 * See the docs for \link kernels/subtract.hpp subtract() \endlink
 */
const local_range_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                         int, int, int, int, int, int, int, int, int, int, int, int, int >
        neg_binomial_2_log_glm("neg_binomial_2_log_glm", neg_binomial_2_log_glm_kernel_code, {{"REDUCTION_STEP_SIZE",4},{"LOCAL_SIZE_", 64}});


}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan

#endif
#endif
