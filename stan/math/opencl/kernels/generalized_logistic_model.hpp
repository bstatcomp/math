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
    
    _Pragma ("OPENCL EXTENSION cl_khr_fp64: enable")
    _Pragma ("OPENCL EXTENSION cl_khr_int64_base_atomics: enable")
    inline void atomicAddLocal(__local double *val, const double delta) {
    
      union {
        double f;
        ulong  i;
      } old;
      union {
        double f;
        ulong  i;
      } new;
      do {
        old.f = *val;
        new.f = old.f + delta;
      } while (atom_cmpxchg ( (volatile __local ulong *)val, old.i, new.i) != old.i);
    }

    inline void atomicAddGlobal(__global double *val, const double delta) {
    
      union {
        double f;
        ulong  i;
      } old;
      union {
        double f;
        ulong  i;
      } new;
      do {
        old.f = *val;
        new.f = old.f + delta;
      } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);
    }
    
    __kernel void generalized_logistic_model(__global double *tmp, __global double *IDp,
                           __global double *IDs, __global double *eta_ps, __global double *eta_ss,
                           __global double *eta_pr, __global double *eta_sr,
                           __global double *X_s, __global double *theta_s,
                           __global double *X_r, __global double *theta_r,
                           __global double *time, __global double *is_pbo, __global double *score,
                           __global double *d_eta,
                           __global double *temp_results,
                           int eta_p_size,
                           int eta_s_size) {
        int i = get_global_id(0);
        int N = get_global_size(0);
        int local_i = get_local_id(0);
        __local int idp_offset;
        __local int ids_offset;
        __local int idp_max;
        __local int ids_max;

        if (local_i==0){
            idp_offset=IDp[i]-1;
            ids_offset=IDs[i]-1;
        }
        if (local_i==get_local_size(0)-1){
            ids_max=IDs[i];
            idp_max=IDp[i];
        }
        
        __local double d_eta_ps_l[L_MEM_SIZE];
        __local double d_eta_ss_l[L_MEM_SIZE];
        __local double d_eta_pr_l[L_MEM_SIZE];
        __local double d_eta_sr_l[L_MEM_SIZE];
        
        for(int i_eta=local_i;i_eta<L_MEM_SIZE;i_eta+=get_local_size(0)){
            d_eta_ps_l[i_eta]=0.0;
            d_eta_ss_l[i_eta]=0.0;
            d_eta_pr_l[i_eta]=0.0;
            d_eta_sr_l[i_eta]=0.0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
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
        double tgt;
        if ((muS * tmp[3]>=0) && (!isinf(muS * tmp[3])) && ((1 - muS) * tmp[3] >=0))
            tgt = dbeta(score[i], muS * tmp[3], (1 - muS) * tmp[3]);
        else
            tgt = INFINITY;
        //double tgt = dbeta(score[i], muS * tmp[3], (1 - muS) * tmp[3]);
        temp_results[i] = tgt;
        temp_results[i+N] = tmp_s;
        temp_results[i+2*N] = tmp_r;
        temp_results[i+3*N] = d_tau;
        temp_results[i+4*N] = d_x_d_mu * (-is_pbo[ids - 1]) * temp1 * temp2;
        temp_results[i+5*N] = d_x_d_mu * (-is_pbo[ids - 1] * tmp[5])
             * ((-tmp[6] / ((tmp[7] - tmp[6]) * (tmp[7] - tmp[6]))) * temp2 + temp1 * exp(-tmp[7] * time[i]) * time[i]);
        temp_results[i+6*N] = d_x_d_mu * (-is_pbo[ids - 1] * tmp[5])
             * ((tmp[7] / ((tmp[7] - tmp[6]) * (tmp[7] - tmp[6]))) * temp2 - temp1 * exp(-tmp[6] * time[i]) * time[i]);
        temp_results[i+7*N] = d_beta;

        for(int j=0;j<tmp[12];j++) {
            int indxr = j*tmp[13]+i;
            int indtmp = i+(8+j)*N;
            temp_results[indtmp] = tmp_r*X_r[indxr];
        }
        for(int j=0;j<tmp[10];j++) {
            int indxs = j*tmp[11]+i;
            int indtmp = i+(8+tmp[12]+j)*N;
            temp_results[indtmp] = tmp_s*X_s[indxs];
        }
        
        //printf("access: %d %f %d\n",i , tgt, );
        atomicAddLocal(&d_eta_ps_l[-idp_offset+idp-1], tmp_s);
        atomicAddLocal(&d_eta_ss_l[-ids_offset+ids-1], tmp_s);
        atomicAddLocal(&d_eta_pr_l[-idp_offset+idp-1], tmp_r);
        atomicAddLocal(&d_eta_sr_l[-ids_offset+ids-1], tmp_r);
        
        barrier(CLK_LOCAL_MEM_FENCE);
        if (local_i<L_MEM_SIZE){
            double val1=d_eta_pr_l[local_i];
            double val2=d_eta_ps_l[local_i];  
            atomicAddGlobal(&d_eta[idp_offset+local_i], val1);
            atomicAddGlobal(&d_eta[eta_p_size+eta_s_size+idp_offset+local_i], val2);

            val1=d_eta_sr_l[local_i];
            val2=d_eta_ss_l[local_i];
            atomicAddGlobal(&d_eta[ids_offset+eta_p_size+local_i], val1);
            atomicAddGlobal(&d_eta[ids_offset+2*eta_p_size+eta_s_size+local_i], val2);
        }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/generalized_logistic_model.hpp generalized_logistic_model() \endlink
 */
const kernel_cl<in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, in_buffer, out_buffer, out_buffer, int, int>
    generalized_logistic_model("generalized_logistic_model", {indexing_helpers, lbeta_helpers, generalized_logistic_model_kernel_code});

// \cond
static const char *reduce_rows_kernel_code = STRINGIFY(
    // \endcond
    /**
     * This kernel peforms a sum reduce on each row of the matrix
     * 
     * @param[in] data input matrix of values to reduce sum
     * @param[out] result a vector of reduced rows
     * @param N number of columns in each row
     * 
     */
    
    __kernel void reduce_rows(
            __global double *data,
            __global double *result,
            unsigned int N) {
    int i = get_global_id(0)%NUM;
    int id = get_global_id(0)/NUM;
    __local double sum[NUM];  
    sum[i] = 0.0;
    for( int j=0;j<N;j+=NUM) {
        if((i+j)<N) {
            sum[i] += data[j+i+id*N];
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
 * See the docs for \link kernels/generalized_logistic_model.hpp reduce_rows() \endlink
 */
const kernel_cl<in_buffer, out_buffer, int>
    reduce_rows("reduce_rows", {indexing_helpers, reduce_rows_kernel_code});


// \cond
static const char *reduce_d_eta_kernel_code = STRINGIFY(
    // \endcond
    /**
     * This kernel peforms a specific reduce of the gradients for
     * the eta parameters for the generalized logistic model.
     * It computes the values to reduce on the fly from IDp,
     * IDs and the precomputed tmp_r and tmp_s values.
     * This is done in order to reduce the memory consumption.
     * 
     * @param[out] d_eta gradients of the eta parameters
     * @param[in] the temporary results for all iterations for the gradients and intermediate results
     * @param[in] IDp vector of patient IDs
     * @param[in] IDs vector of study IDs
     * @param N number of columns in each row
     * @param d_eta_p_size size of eta_p vectors
     * @param d_eta_s_size size of eta_s vectors
     */
    __kernel void reduce_d_eta(
            __global double *d_eta,
            __global double *temp_results,
            __global double *IDp,
            __global double *IDs,
            unsigned int N,
            unsigned int d_eta_p_size,
            unsigned int d_eta_s_size) {
        const int i = get_global_id(0)%NUM;
        const int id = get_global_id(0)/NUM;
        __local double sum[NUM];  
        sum[i] = 0.0;
        if(id<d_eta_p_size) {
            for( int j=0;j<N;j+=NUM) {
                if((i+j)<N) {
                     if((IDp[i+j]-1)==id){
                        sum[i] += temp_results[i+j+2*N];
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
                d_eta[id] = sum[0];
            }            
        }else if(id>=d_eta_p_size && id < (d_eta_p_size+d_eta_s_size)) {
            const int idr = id - d_eta_p_size;
            for( int j=0;j<N;j+=NUM) {
                if((i+j)<N) {
                    if((IDs[i+j]-1)==idr){
                        sum[i] += temp_results[i+j+2*N];
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
                d_eta[id] = sum[0];
            }
        }else if(id>=(d_eta_p_size+d_eta_s_size) && id< (2*d_eta_p_size+d_eta_s_size)) {
            const int ida = id - d_eta_p_size - d_eta_s_size;
            for( int j=0;j<N;j+=NUM) {
                if((i+j)<N) {
                    if((IDp[i+j]-1)==ida){
                        sum[i] += temp_results[i+j+N];
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
                d_eta[id] = sum[0];
            }            
            
        }else {
            const int ids = id - 2*d_eta_p_size-d_eta_s_size;
            for( int j=0;j<N;j+=NUM) {
                if((i+j)<N) {
                    if((IDs[i+j]-1)==ids){
                        sum[i] += temp_results[i+j+N];
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
                d_eta[id] = sum[0];
            }
        }
    }
    // \cond
);
// \endcond

/**
 * See the docs for \link kernels/generalized_logistic_model.hpp reduce_d_eta() \endlink
 */
const kernel_cl<out_buffer, in_buffer, in_buffer, in_buffer, int, int, int>
    reduce_d_eta("reduce_d_eta",
             {indexing_helpers, reduce_d_eta_kernel_code});


}  // namespace opencl_kernels
}  // namespace math
}  // namespace stan
#endif
#endif
