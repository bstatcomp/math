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
    double digamma(double x)
    {
    double c = 8.5;
    double euler_mascheroni = 0.57721566490153286060;
    double r;
    double value;
    double x2;
    if (x <= 0.0)
    {
        value = 0.0;
        return value;
    }
    if (x <= 0.000001)
    {
        value = -euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x;
        return value;
    }
    value = 0.0;
    x2 = x;
    while (x2 < c)
    {
        value = value - 1.0 / x2;
        x2 = x2 + 1.0;
    }
    r = 1.0 / x2;
    value = value + log(x2) - 0.5 * r;

    r = r * r;
    
    value = value
        - r * (1.0 / 12.0
        - r * (1.0 / 120.0
            - r * (1.0 / 252.0
            - r * (1.0 / 240.0
                - r * (1.0 / 132.0)))));

    return value;
    }  
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
                           __global double *X_s, __global double *theta_s,
                           __global double *X_r, __global double *theta_r,
                           __global double *time, __global double *is_pbo, __global double *score,
                           __global double *outtmp, __global double *outtmp1, __global double *outtmp2,
                           __global double *outtmp3, __global double *outtmp4,__global double *outtmp5,
                           __global double *outtmp6, __global double *outtmp7, __global double *outtmp8) {
        int i = get_global_id(0);
        int idp = IDp[i];
        int ids = IDs[i];
        double cov_s = tmp[8] + eta_ps[idp-1] + eta_ss[ids-1];
        double cov_r = tmp[9] + eta_pr[idp - 1] + eta_sr[ids - 1];
        int j = 0;
        if(tmp[14]*tmp[15]>0){
            for(j=0;j<tmp[10];j++) {
            int id_xs = j*tmp[2]+i;
            int id_theta_s = j;
            cov_s = cov_s + X_s[id_xs]*theta_s[id_theta_s];
            }
        }
        if(tmp[16]*tmp[17]>0){
            for(j=0;j<tmp[12];j++) {
            int id_xr = j*tmp[2]+i;
            int id_theta_r = j;
            cov_r = cov_r + X_r[id_xr]*theta_r[id_theta_r];
            }
        }

        if (tmp[0] == 1) {
            cov_s = exp(cov_s);
        }      
        if (tmp[1] == 1) {
            cov_r = exp(cov_r);
        }  

        double temp1 = tmp[7] / (tmp[7] - tmp[6]);
        double temp2 = (exp(-tmp[6] * time[i]) - exp(-tmp[7] * time[i]));
        double S0 = 1 / (1 + exp(-cov_s));
        double pbo_eff = tmp[5] * temp1 * temp2;
        double S0_beta_pow = pow(S0, tmp[4]);
        double exp10 = exp(-tmp[4] * cov_r * time[i]);
        double muS = S0
                  / pow((S0_beta_pow + (1 - S0_beta_pow) * exp10), 1.0 / tmp[4])
                  - is_pbo[ids-1] * pbo_eff;
        double d_x_d_mu = tmp[3] * log(score[i]) - tmp[3] * log(1 - score[i]) - digamma(muS * tmp[3]) * tmp[3] + digamma(tmp[3] - muS * tmp[3]) * tmp[3];
    
    
        //const double temp1 = outtmp3(0,i);//k_eq / (k_eq - k_el);
        //const double temp2 = outtmp4(0,i);//(exp(-k_el * time[i]) - exp(-k_eq * time[i]));
        //const double pbo_eff = outtmp5(0,i);//beta_pbo * temp1 * temp2;
        outtmp[i] = cov_s;
        outtmp1[i] = cov_r;
        outtmp2[i] = S0;
        outtmp3[i] = temp1;
        outtmp4[i] = temp2;
        outtmp5[i] = d_x_d_mu;
        outtmp6[i] = S0_beta_pow;
        outtmp7[i] = exp10;
        outtmp8[i] = muS;
      
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/subtract.hpp subtract() \endlink
 */
const kernel_cl<in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer,
                out_buffer, out_buffer, out_buffer, out_buffer, out_buffer, out_buffer, out_buffer, out_buffer, out_buffer>
    generalized_logistic_model("generalized_logistic_model",
             {indexing_helpers, generalized_logistic_model_kernel_code});

}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
