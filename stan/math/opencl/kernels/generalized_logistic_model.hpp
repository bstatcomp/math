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
    double dbeta(const double x, const double a, const double b) {
        return log(x) * (a - 1) + log(1 - x) * (b - 1) - lbeta(a, b);
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
                           __global double *outtmp2,
                           __global double *outtmp3, __global double *outtmp4,
                           __global double *outtmp6, __global double *outtmp9,
                           __global double *outtmp10, __global double *outtmp11, __global double *outtmp12) {
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
        double d_tau = muS * log(score[i]) + log(1 - score[i]) - muS * log(1 - score[i]) - digamma(muS * tmp[3]) * muS - digamma(tmp[3] - muS * tmp[3]) * (1 - muS) + digamma(tmp[3]);

        const double temp9 = pow(S0, tmp[4]);    
        const double alpha = pow(temp9 + (1 - temp9) * exp10, 1.0 / tmp[4]);
        const double alpha_sq = (alpha * alpha);
        const double temp12 = 1.0 / tmp[4] * alpha * pow(alpha, -tmp[4]);
        double tmp_s = d_x_d_mu
                   * ((alpha - S0 * temp12
                      * (tmp[4] * pow(S0, tmp[4] - 1) - exp10 * tmp[4] * pow(S0, tmp[4] - 1)))
                      / alpha_sq)
                   * (exp(-cov_s) / ((1 + exp(-cov_s)) * (1 + exp(-cov_s))));
        double tmp_r = d_x_d_mu * S0 * (-(temp12 * (-exp10 * tmp[4] * time[i] + temp9 * exp10 * tmp[4] * time[i])) / alpha_sq);
        double alpha_beta_pow = pow(alpha, tmp[4]);
        const double temp13 = temp9 * log(S0);
        const double temp14 = exp10 * cov_r * time[i];
        double d_beta = d_x_d_mu * (-S0 * pow(alpha, -2)) * alpha
              * ((log(alpha_beta_pow) * (-pow(tmp[4], -2)))
              + (1.0 / tmp[4] * (temp13 - temp14 - temp13 * exp10 + temp9 * temp14))
              / alpha_beta_pow);
        if (tmp[0] == 1) {
            tmp_s = tmp_s * cov_s;
        }      
        if (tmp[1] == 1) {
            tmp_r = tmp_r * cov_r;
        }
        double tgt = dbeta(score[i], muS * tmp[3], (1 - muS) * tmp[3]);
        outtmp2[i] = tgt;
        outtmp3[i] = tmp_s;
        outtmp4[i] = tmp_r;
        outtmp6[i] = d_tau;
        outtmp9[i] = d_x_d_mu * (-is_pbo[ids - 1]) * temp1 * temp2;
        outtmp10[i] = d_x_d_mu * (-is_pbo[ids - 1] * tmp[5])
             * ((-tmp[6] / ((tmp[7] - tmp[6]) * (tmp[7] - tmp[6]))) * temp2 + temp1 * exp(-tmp[7] * time[i]) * time[i]);
        outtmp11[i] = d_x_d_mu * (-is_pbo[ids - 1] * tmp[5])
             * ((tmp[7] / ((tmp[7] - tmp[6]) * (tmp[7] - tmp[6]))) * temp2 - temp1 * exp(-tmp[6] * time[i]) * time[i]);
        outtmp12[i] = d_beta;
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/subtract.hpp subtract() \endlink
 */
const kernel_cl<in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer,
                out_buffer, out_buffer, out_buffer, out_buffer, out_buffer, out_buffer, out_buffer, out_buffer>
    generalized_logistic_model("generalized_logistic_model",
             {indexing_helpers, helpnow, generalized_logistic_model_kernel_code});
// \cond
static const char *reduce_kernel_code = STRINGIFY(
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
    
    __kernel void reduce(
            __global double *in1,
            __global double *in2,
            __global double *in3,
            __global double *in4,
            __global double *in5,
            __global double *in6,
            __global double *tmp_s,
            __global double *tmp_r,
            __global double *result,
            unsigned int N) {
    int i = get_global_id(0)%NUM;
    int id = get_global_id(0)/NUM;
    __local double sum[NUM];  
    sum[i] = 0.0;
    for( int j=0;j<N;j+=NUM) {
        if((i+j)<N) {
            if(id==0){
                sum[i] += in1[j+i];
            }
            if(id==1){
                sum[i] += in2[j+i];
            }
            if(id==2){
                sum[i] += in3[j+i];
            }
            if(id==3){
                sum[i] += in4[j+i];
            }
            if(id==4){
                sum[i] += in5[j+i];
            }
            if(id==5){
                sum[i] += in6[j+i];
            }
            if(id==6){
                sum[i] += tmp_s[j+i];
            }
            if(id==7){
                sum[i] += tmp_r[j+i];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i<NUM2){
        for( int j=NUM2;j<NUM;j+=NUM2) {
            if((i+j)<NUM) {  
                sum[i] += sum[i+j];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i<1){
        for( int j=1;j<NUM2;j++) {
            sum[0] += sum[j];
        }
        result[id] = sum[0];
    }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/subtract.hpp subtract() \endlink
 */
const kernel_cl<in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, out_buffer, int>
    reduce("reduce",
             {indexing_helpers, reduce_kernel_code});

// \cond
static const char *reduce2_kernel_code = STRINGIFY(
    // \endcond
    
    __kernel void reduce2(
            __global double *d_theta_s,
            __global double *d_theta_r,
            __global double *tmp_s,
            __global double *tmp_r,
            __global double *X_s,
            __global double *X_r,
            unsigned int N,
            unsigned int X_s_rows,
            unsigned int X_s_cols,
            unsigned int X_r_rows,
            unsigned int X_r_cols) {
    const int i = get_global_id(0)%NUM;
    const int id = get_global_id(0)/NUM;
    const int id_r = id - X_s_cols;
    __local double sum[NUM];  
    sum[i] = 0.0;
    for( int j=0;j<N;j+=NUM) {
        if((i+j)<N) {
            if(id < X_s_cols){
                sum[i] += tmp_s[j+i]*X_s[id*X_s_rows+j+i];
            }else{
                
                sum[i] += tmp_r[j+i]*X_r[id_r*X_r_rows+j+i];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i<NUM2){
        for( int j=NUM2;j<NUM;j+=NUM2) {
            if((i+j)<NUM) {  
                sum[i] += sum[i+j];
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i<1){
        for( int j=1;j<NUM2;j++) {
            sum[0] += sum[j];
        }
        if(id < X_s_cols){
            d_theta_s[id] = sum[0];
        }else{
            d_theta_r[id_r] = sum[0];
        }
    }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/subtract.hpp subtract() \endlink
 */
const kernel_cl<out_buffer, out_buffer, in_buffer, in_buffer, in_buffer, in_buffer, int, int, int, int, int>
    reduce2("reduce2",
             {indexing_helpers, reduce2_kernel_code});

// \cond
static const char *reduce3_kernel_code = STRINGIFY(
    // \endcond
    
    __kernel void reduce3(
            __global double *d_eta_ps,
            __global double *d_eta_ss,
            __global double *d_eta_pr,
            __global double *d_eta_sr,
            __global double *tmp_s,
            __global double *tmp_r,
            __global double *IDp,
            __global double *IDs,
            unsigned int N,
            unsigned int d_eta_p_size,
            unsigned int d_eta_s_size) {
        const int id = get_global_id(0);
        if(id<d_eta_p_size) {
            double sum = 0.0;
            for(int i=0;i<N;i++) {
                if((IDp[i]-1)==id){
                    sum += tmp_s[i];
                }
            }
            d_eta_ps[id] = sum;
        }else if(id>=d_eta_p_size && id < 2*d_eta_p_size) {
            const int idp = id -d_eta_p_size;            
            double sum = 0.0;
            for(int i=0;i<N;i++) {
                if((IDp[i]-1)==idp){
                    sum += tmp_r[i];
                }
            }
            d_eta_pr[idp] = sum;
        }else if(id>=(2*d_eta_p_size) && id< (2*d_eta_p_size+d_eta_s_size)) {
            const int ids = id - 2*d_eta_p_size;
            double sum = 0.0;
            for(int i=0;i<N;i++) {
                if((IDs[i]-1)==ids){
                    sum += tmp_s[i];
                }
            }
            d_eta_ss[ids] = sum;
        }else {
            const int idr = id - 2*d_eta_p_size - d_eta_s_size;
            double sum = 0.0;
            for(int i=0;i<N;i++) {
                if((IDs[i]-1)==idr){
                    sum += tmp_r[i];
                }
            }
            d_eta_sr[idr] = sum;
        }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/subtract.hpp subtract() \endlink
 */
const kernel_cl<out_buffer, out_buffer, out_buffer, out_buffer, in_buffer, in_buffer, in_buffer, in_buffer, int, int, int>
    reduce3("reduce3",
             {indexing_helpers, reduce3_kernel_code});


}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
