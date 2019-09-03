#ifndef STAN_MATH_REV_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP
#define STAN_MATH_REV_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP

#ifdef STAN_OPENCL
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/kernels/generalized_logistic_model.hpp>
#include <stan/math/opencl/err/check_matching_dims.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#endif
#include <stan/math/rev/core.hpp>
#include <algorithm>

namespace stan {
namespace math {
#ifdef STAN_OPENCL

int copied = 0;
cl::Buffer IDp_buf;
cl::Buffer IDs_buf;
cl::Buffer X_s_buf;
cl::Buffer X_r_buf;
cl::Buffer is_pbo_buf;
cl::Buffer score_buf;
cl::Buffer time_buf;

inline var generalized_logistic_model(
    const std::vector<int>& IDp, const std::vector<int>& IDs,
    const std::vector<int>& is_pbo, const vector_d& time, const vector_d& score,
    const int multiplicative_s, const int multiplicative_r,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_s,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_r,
    var& tauv, var& betav, var& beta_pbov,
    var& k_elv, var& k_eqv,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& theta_r,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& theta_s,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_pr,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_sr,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_ps,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_ss,
    var& base_sv, var& base_rv) {
  const int N = IDp.size();
  const double tau = tauv.val();
  const double beta = betav.val();
  const double beta_pbo = beta_pbov.val();
  const double k_el = k_elv.val();
  const double k_eq = k_eqv.val();
  const double base_s = base_sv.val();
  const double base_r = base_rv.val();

  double d_tau = 0;
  double d_beta_pbo = 0;
  double d_k_eq = 0;
  double d_k_el = 0;
  double d_beta = 0;
  double d_base_s = 0;
  double d_base_r = 0;
  
  const int theta_s_size = X_s.cols();
  const int theta_r_size = X_r.cols();
  
  std::vector<double> d_eta_pr(eta_pr.size());
  std::vector<double> d_eta_sr(eta_sr.size());
  std::vector<double> d_eta_ps(eta_ps.size());
  std::vector<double> d_eta_ss(eta_ss.size());
  std::vector<double> d_theta_s(X_s.cols());
  std::vector<double> d_theta_r(X_r.cols());

  const int p_size = d_eta_pr.size();
  const int s_size = d_eta_sr.size();
  matrix_cl<double> IDpp_cl(1, IDp.size());
  if(copied == 0){
    matrix_cl<double> tX_s_cl = to_matrix_cl<double>(X_s);
    tX_s_cl.wait_for_read_write_events();
    X_s_buf = tX_s_cl.buffer();
    matrix_cl<double> tX_r_cl = to_matrix_cl<double>(X_r);
    tX_r_cl.wait_for_read_write_events();
    X_r_buf = tX_r_cl.buffer();

    matrix_d IDp_temp(1, IDp.size());
    matrix_d IDs_temp(1, IDs.size());
    matrix_d is_pbo_temp(is_pbo.size(), 1);
    for(int i=0;i<IDp.size();i++) {
      IDp_temp(i) = static_cast<double>(IDp[i]);
    }
    for(int i=0;i<IDs.size();i++) {
      IDs_temp(i) = static_cast<double>(IDs[i]);
    }
    for(int i=0;i<is_pbo.size();i++) {
      is_pbo_temp(i, 0) = is_pbo[i];
    } 
    matrix_cl<double> tIDp_cl = to_matrix_cl<double>(IDp_temp);
    tIDp_cl.wait_for_read_write_events();
    IDp_buf = tIDp_cl.buffer();
    matrix_cl<double> tIDs_cl = to_matrix_cl<double>(IDs_temp);
    tIDs_cl.wait_for_read_write_events();
    IDs_buf = tIDs_cl.buffer();
    matrix_cl<double> tis_pbo_cl = to_matrix_cl<double>(is_pbo_temp);
    tis_pbo_cl.wait_for_read_write_events();
    is_pbo_buf = tis_pbo_cl.buffer();    
    matrix_cl<double> tscore_cl = to_matrix_cl<double>(score);
    tscore_cl.wait_for_read_write_events();
    score_buf = tscore_cl.buffer();
    matrix_cl<double> ttime_cl = to_matrix_cl<double>(time);
    ttime_cl.wait_for_read_write_events();
    time_buf = ttime_cl.buffer();
    copied == 1;
  }
  matrix_cl<double> IDp_cl(IDp_buf, 1, N);
  matrix_cl<double> IDs_cl(IDs_buf, 1, N);
  matrix_cl<double> X_s_cl(X_s_buf, 1, N);
  matrix_cl<double> X_r_cl(X_r_buf, 1, N);
  matrix_cl<double> time_cl(time_buf, 1, N);
  matrix_cl<double> score_cl(score_buf, 1, N);
  matrix_cl<double> is_pbo_cl(is_pbo_buf, 1, N);
  
  matrix_d t1 = eta_ps.val();
  matrix_d t2 = eta_ss.val();
  matrix_d t3 = eta_pr.val();
  matrix_d t4 = eta_sr.val();
  matrix_d t5 = theta_r.val();
  matrix_d t6 = theta_s.val();
  matrix_cl<double> eta_ps_cl(t1);
  matrix_cl<double> eta_ss_cl(t2);
  matrix_cl<double> eta_pr_cl(t3);
  matrix_cl<double> eta_sr_cl(t4);
  matrix_cl<double> theta_r_cl(t5);
  matrix_cl<double> theta_s_cl(t6);
  matrix_d tmp(18,1);
  tmp(0,0) = multiplicative_s;
  tmp(1,0) = multiplicative_r;
  tmp(2,0) = N;
  tmp(3,0) = tau;
  tmp(4,0) = beta;
  tmp(5,0) = beta_pbo;
  tmp(6,0) = k_el;
  tmp(7,0) = k_eq;
  tmp(8,0) = base_s;
  tmp(9,0) = base_r;
  tmp(10,0) = X_s.cols();
  tmp(11,0) = X_s.rows();
  tmp(12,0) = X_r.cols();
  tmp(13,0) = X_r.rows();
  tmp(14,0) = theta_s.cols();
  tmp(15,0) = theta_s.rows();
  tmp(16,0) = theta_r.cols();
  tmp(17,0) = theta_r.rows();
  matrix_cl<double> tmp_cl(tmp);
  double tgt = 0;

  matrix_cl<double> cov_s_cl(tmp);
  matrix_cl<double> cov_r_cl(tmp);
  matrix_cl<double> outtmp_cl(1, N);
  matrix_cl<double> outtmp1_cl(1, N);
  matrix_cl<double> outtmp2_cl(1, N);
  matrix_cl<double> outtmp3_cl(1, N);
  matrix_cl<double> outtmp4_cl(1, N);
  matrix_cl<double> outtmp5_cl(1, N);
  matrix_cl<double> outtmp6_cl(1, N);
  matrix_cl<double> outtmp7_cl(1, N);
  matrix_cl<double> outtmp8_cl(1, N);
  matrix_d cov_s_tmp(N,1);
  matrix_d cov_r_tmp(N,1); 
  matrix_d outtmp(1, N);
  matrix_d outtmp1(1, N);
  matrix_d outtmp2(1, N);
  matrix_d outtmp3(1, N);
  matrix_d outtmp4(1, N);
  matrix_d outtmp5(1, N);
  matrix_d outtmp6(1, N);
  matrix_d outtmp7(1, N);
  matrix_d outtmp8(1, N);
  try {
    opencl_kernels::generalized_logistic_model(cl::NDRange(N), tmp_cl, IDp_cl, IDs_cl, eta_ps_cl, eta_ss_cl, eta_pr_cl, eta_sr_cl, X_s_cl, theta_s_cl, X_r_cl, theta_r_cl, time_cl, is_pbo_cl, score_cl, outtmp_cl, outtmp1_cl, outtmp2_cl, outtmp3_cl, outtmp4_cl, outtmp5_cl, outtmp6_cl, outtmp7_cl, outtmp8_cl);
  } catch (cl::Error& e) {
    check_opencl_error("generalized_logistic_model", e);
  }

  outtmp = from_matrix_cl(outtmp_cl);
  outtmp1 = from_matrix_cl(outtmp1_cl);
  outtmp2 = from_matrix_cl(outtmp2_cl);
  outtmp3 = from_matrix_cl(outtmp3_cl);
  outtmp4 = from_matrix_cl(outtmp4_cl);
  outtmp5 =  from_matrix_cl(outtmp5_cl);
  outtmp6 = from_matrix_cl(outtmp6_cl);
  outtmp7 = from_matrix_cl(outtmp7_cl);
  outtmp8 = from_matrix_cl(outtmp8_cl);
  for (int i = 0; i < N; i++) {
    // compute function
    double cov_r = outtmp1(0,i);
    double cov_s = outtmp(0,i);
    const double S0 = outtmp2(0,i);
    const double temp1 = outtmp3(0,i);
    const double temp2 = outtmp4(0,i);
    const double pbo_eff = outtmp5(0,i);
    const double S0_beta_pow = outtmp6(0,i);
    const double temp10 = outtmp7(0,i);
    double muS = outtmp8(0,i);

    // compute gradients
    const double temp3 = digamma(muS * tau);
    const double temp4 = digamma(tau - muS * tau);
    const double temp5 = log(score[i]);
    const double temp6 = log(1 - score[i]);
    double d_x_d_mu = tau * temp5 - tau * temp6 - temp3 * tau + temp4 * tau;
    d_tau = d_tau + muS * temp5 + temp6 - muS * temp6 - temp3 * muS - temp4 * (1 - muS) + digamma(tau);
    const double temp8 =  ((k_eq - k_el) * (k_eq - k_el));

    d_beta_pbo = d_beta_pbo + d_x_d_mu * (-is_pbo[IDs[i] - 1]) * temp1 * temp2;
    d_k_eq = d_k_eq + d_x_d_mu * (-is_pbo[IDs[i] - 1] * beta_pbo)
             * ((-k_el / temp8) * temp2 + temp1 * exp(-k_eq * time[i]) * time[i]);
    d_k_el = d_k_el + d_x_d_mu * (-1*is_pbo[IDs[i] - 1] * beta_pbo)
             * ((k_eq / temp8) * temp2 - temp1 * exp(-k_el * time[i]) * time[i]);

    const double temp9 = std::pow(S0, beta);
    const double exp10 = exp(temp10);
    const double alpha = std::pow(temp9 + (1 - temp9) * exp10, 1.0 / beta);
    const double temp11 = std::pow(S0, beta - 1);
    const double alpha_sq = (alpha * alpha);
    const double temp12 = 1.0 / beta * alpha * std::pow(alpha, -beta);
    const double exp_neg_cov_s = exp(-cov_s);
    double tmp_s = d_x_d_mu
                   * ((alpha - S0 * temp12
                      * (beta * temp11 - exp10 * beta * temp11))
                      / alpha_sq)
                   * (exp_neg_cov_s / ((1 + exp_neg_cov_s) * (1 + exp_neg_cov_s)));
    const double temp15 = exp10 * beta * time[i];
    double tmp_r = d_x_d_mu * S0 * (-(temp12 * (-temp15 + temp9 * temp15)) / alpha_sq);
    if (multiplicative_s == 1) {
      tmp_s = tmp_s * cov_s;
    }      
    if (multiplicative_r == 1) {
      tmp_r = tmp_r * cov_r;
    }  

    d_base_s += tmp_s;
    
    for (int c = 0; c < X_s.cols(); c++) {
      d_theta_s[c] += tmp_s * X_s(i, c);
    }      

    d_eta_ps[IDp[i] - 1] = d_eta_ps[IDp[i] - 1] + tmp_s;
    d_eta_ss[IDs[i] - 1] = d_eta_ss[IDs[i] - 1] + tmp_s;

    d_base_r += tmp_r;
    for (int c = 0; c < X_r.cols(); c++) {
      d_theta_r[c] += tmp_r * X_r(i, c);
    }     

    d_eta_pr[IDp[i] - 1] = d_eta_pr[IDp[i] - 1] + tmp_r;
    d_eta_sr[IDs[i] - 1] = d_eta_sr[IDs[i] - 1] + tmp_r;
    const double alpha_beta_pow = std::pow(alpha, beta);
    const double temp13 = temp9 * log(S0);
    const double temp14 = exp10 * cov_r * time[i];
    d_beta += d_x_d_mu * (-S0 * std::pow(alpha, -2)) * alpha
              * ((log(alpha_beta_pow) * (-std::pow(beta, -2)))
              + (1.0 / beta * (temp13 - temp14 - temp13 * exp10 + temp9 * temp14))
              / alpha_beta_pow);
    tgt += dbeta(score[i], muS * tau, (1 - muS) * tau);
  }

  // stack it up
  vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
      7 + theta_r_size + theta_s_size + p_size + p_size + s_size + s_size);
  varis[0] = tauv.vi_;
  varis[1] = betav.vi_;
  varis[2] = beta_pbov.vi_;
  varis[3] = k_elv.vi_;
  varis[4] = k_eqv.vi_;
  varis[5] = base_sv.vi_;
  varis[6] = base_rv.vi_;

  int k = 7;
  for (int i = 0; i < theta_r.size(); i++) {
    varis[k] = theta_r(i).vi_;
    k++;
  }
  for (int i = 0; i < theta_s.size(); i++) {
    varis[k] = theta_s(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_pr.size(); i++) {
    varis[k] = eta_pr(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_sr.size(); i++) {
    varis[k] = eta_sr(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_ps.size(); i++) {
    varis[k] = eta_ps(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_ss.size(); i++) {
    varis[k] = eta_ss(i).vi_;
    k++;
  }
  double* gradients = ChainableStack::instance_->memalloc_.alloc_array<double>(
      7 + theta_r_size + theta_s_size + p_size + p_size + s_size + s_size);
  gradients[0] = d_tau;
  gradients[1] = d_beta;
  gradients[2] = d_beta_pbo;
  gradients[3] = d_k_el;
  gradients[4] = d_k_eq;
  gradients[5] = d_base_s;
  gradients[6] = d_base_r;

  k = 7;
  for (int i = 0; i < theta_r.size(); i++) {
    gradients[k] = d_theta_r[i];
    k++;
  }
  for (int i = 0; i < theta_s.size(); i++) {
    gradients[k] = d_theta_s[i];
    k++;
  }
  for (int i = 0; i < p_size; i++) {
    gradients[k] = d_eta_pr[i];
    k++;
  }
  for (int i = 0; i < s_size; i++) {
    gradients[k] = d_eta_sr[i];
    k++;
  }
  for (int i = 0; i < p_size; i++) {
    gradients[k] = d_eta_ps[i];
    k++;
  }
  for (int i = 0; i < s_size; i++) {
    gradients[k] = d_eta_ss[i];
    k++;
  }
  return var(new precomputed_gradients_vari(
      tgt, 7 + theta_r_size + theta_s_size + p_size + p_size + s_size + s_size,
      varis, gradients));
}
#else
/*
inline var generalized_logistic_model(
    const std::vector<int>& IDp, const std::vector<int>& IDs,
    const std::vector<int>& is_pbo, const vector_d& time, const vector_d& score,
    const int multiplicative_s, const int multiplicative_r,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_s,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_r,
    var& tauv, var& betav, var& beta_pbov,
    var& k_elv, var& k_eqv,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& theta_r,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& theta_s,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_pr,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_sr,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_ps,
    const Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic>& eta_ss,
    var& base_sv, var& base_rv) {
  const int N = IDp.size();
  const double tau = tauv.val();
  const double beta = betav.val();
  const double beta_pbo = beta_pbov.val();
  const double k_el = k_elv.val();
  const double k_eq = k_eqv.val();
  const double base_s = base_sv.val();
  const double base_r = base_rv.val();

  double d_tau = 0;
  double d_beta_pbo = 0;
  double d_k_eq = 0;
  double d_k_el = 0;
  double d_beta = 0;
  double d_base_s = 0;
  double d_base_r = 0;
  
  const int theta_s_size = X_s.cols();
  const int theta_r_size = X_r.cols();
  
  std::vector<double> d_eta_pr(eta_pr.size());
  std::vector<double> d_eta_sr(eta_sr.size());
  std::vector<double> d_eta_ps(eta_ps.size());
  std::vector<double> d_eta_ss(eta_ss.size());
  std::vector<double> d_theta_s(X_s.cols());
  std::vector<double> d_theta_r(X_r.cols());

  const int p_size = d_eta_pr.size();
  const int s_size = d_eta_sr.size();

  double tgt = 0;
  for (int i = 0; i < N; i++) {
    // compute function
    double cov_s = base_s + eta_ps(IDp[i] - 1).val() + eta_ss(IDs[i] - 1).val();
    double cov_r = base_r + eta_pr(IDp[i] - 1).val() + eta_sr(IDs[i] - 1).val();
    if (theta_s.size() > 0)
      cov_s = cov_s + (X_s.row(i) * theta_s.col(0))(0, 0).val();
    if (theta_r.size() > 0) {
      cov_r = cov_r + (X_r.row(i) * theta_r.col(0))(0, 0).val();
    }      
    if (multiplicative_s == 1) {
      cov_s = exp(cov_s);
    }      
    if (multiplicative_r == 1) {
      cov_s = exp(cov_r);
    }      
    const double S0 = 1 / (1 + exp(-cov_s));
    const double temp1 = k_eq / (k_eq - k_el);
    const double temp2 = (exp(-k_el * time[i]) - exp(-k_eq * time[i]));
    const double pbo_eff = beta_pbo * temp1 * temp2;
    const double temp7 = is_pbo[IDs[i] - 1];
    const double inv_beta = 1.0 / beta;
    const double S0_beta_pow = std::pow(S0, beta);
    const double temp10 = -beta * cov_r * time[i];
    double muS = S0
                  / std::pow((S0_beta_pow + (1 - S0_beta_pow) * exp(temp10)), inv_beta)
                  - temp7 * pbo_eff;

    // compute gradients
    const double muS_tau_prod = muS * tau;
    const double temp3 = digamma(muS_tau_prod);
    const double temp4 = digamma(tau - muS_tau_prod);
    const double temp5 = log(score[i]);
    const double temp6 = log(1 - score[i]);
    double d_x_d_mu = tau * temp5 - tau * temp6 - temp3 * tau + temp4 * tau;
    d_tau = d_tau + muS * temp5 + temp6 - muS * temp6 - temp3 * muS - temp4 * (1 - muS) + digamma(tau);
    const double temp8 =  ((k_eq - k_el) * (k_eq - k_el));

    d_beta_pbo = d_beta_pbo + d_x_d_mu * (-temp7) * temp1 * temp2;
    d_k_eq = d_k_eq + d_x_d_mu * (-temp7 * beta_pbo)
             * ((-k_el / temp8) * temp2 + temp1 * exp(-k_eq * time[i]) * time[i]);
    d_k_el = d_k_el + d_x_d_mu * (-1*temp7 * beta_pbo)
             * ((k_eq / temp8) * temp2 - temp1 * exp(-k_el * time[i]) * time[i]);

    const double temp9 = std::pow(S0, beta);
    const double exp10 = exp(temp10);
    const double alpha = std::pow(temp9 + (1 - temp9) * exp10, inv_beta);
    const double temp11 = std::pow(S0, beta - 1);
    const double alpha_sq = (alpha * alpha);
    const double temp12 = inv_beta * alpha * std::pow(alpha, -beta);
    const double exp_neg_cov_s = exp(-cov_s);
    double tmp_s = d_x_d_mu
                   * ((alpha - S0 * temp12
                      * (beta * temp11 - exp10 * beta * temp11))
                      / alpha_sq)
                   * (exp_neg_cov_s / ((1 + exp_neg_cov_s) * (1 + exp_neg_cov_s)));
    const double temp15 = exp10 * beta * time[i];
    double tmp_r = d_x_d_mu * S0 * (-(temp12 * (-temp15 + temp9 * temp15)) / alpha_sq);
    if (multiplicative_s == 1) {
      tmp_s = tmp_s * cov_s;
    }      
    if (multiplicative_r == 1) {
      tmp_r = tmp_r * cov_r;
    }  

    d_base_s += tmp_s;
    
    for (int c = 0; c < X_s.cols(); c++) {
      d_theta_s[c] += tmp_s * X_s(i, c);
    }      

    d_eta_ps[IDp[i] - 1] = d_eta_ps[IDp[i] - 1] + tmp_s;
    d_eta_ss[IDs[i] - 1] = d_eta_ss[IDs[i] - 1] + tmp_s;

    d_base_r += tmp_r;
    for (int c = 0; c < X_r.cols(); c++) {
      d_theta_r[c] += tmp_r * X_r(i, c);
    }     

    d_eta_pr[IDp[i] - 1] = d_eta_pr[IDp[i] - 1] + tmp_r;
    d_eta_sr[IDs[i] - 1] = d_eta_sr[IDs[i] - 1] + tmp_r;
    const double alpha_beta_pow = std::pow(alpha, beta);
    const double temp13 = temp9 * log(S0);
    const double temp14 = exp10 * cov_r * time[i];
    d_beta += d_x_d_mu * (-S0 * std::pow(alpha, -2)) * alpha
              * ((log(alpha_beta_pow) * (-std::pow(beta, -2)))
              + (inv_beta * (temp13 - temp14 - temp13 * exp10 + temp9 * temp14))
              / alpha_beta_pow);
    tgt += dbeta(score[i], muS_tau_prod, (1 - muS) * tau);
  }

  // stack it up
  vari** varis = ChainableStack::instance_->memalloc_.alloc_array<vari*>(
      7 + theta_r_size + theta_s_size + p_size + p_size + s_size + s_size);
  varis[0] = tauv.vi_;
  varis[1] = betav.vi_;
  varis[2] = beta_pbov.vi_;
  varis[3] = k_elv.vi_;
  varis[4] = k_eqv.vi_;
  varis[5] = base_sv.vi_;
  varis[6] = base_rv.vi_;

  int k = 7;
  for (int i = 0; i < theta_r.size(); i++) {
    varis[k] = theta_r(i).vi_;
    k++;
  }
  for (int i = 0; i < theta_s.size(); i++) {
    varis[k] = theta_s(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_pr.size(); i++) {
    varis[k] = eta_pr(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_sr.size(); i++) {
    varis[k] = eta_sr(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_ps.size(); i++) {
    varis[k] = eta_ps(i).vi_;
    k++;
  }
  for (int i = 0; i < eta_ss.size(); i++) {
    varis[k] = eta_ss(i).vi_;
    k++;
  }
  double* gradients = ChainableStack::instance_->memalloc_.alloc_array<double>(
      7 + theta_r_size + theta_s_size + p_size + p_size + s_size + s_size);
  gradients[0] = d_tau;
  gradients[1] = d_beta;
  gradients[2] = d_beta_pbo;
  gradients[3] = d_k_el;
  gradients[4] = d_k_eq;
  gradients[5] = d_base_s;
  gradients[6] = d_base_r;

  k = 7;
  for (int i = 0; i < theta_r.size(); i++) {
    gradients[k] = d_theta_r[i];
    k++;
  }
  for (int i = 0; i < theta_s.size(); i++) {
    gradients[k] = d_theta_s[i];
    k++;
  }
  for (int i = 0; i < p_size; i++) {
    gradients[k] = d_eta_pr[i];
    k++;
  }
  for (int i = 0; i < s_size; i++) {
    gradients[k] = d_eta_sr[i];
    k++;
  }
  for (int i = 0; i < p_size; i++) {
    gradients[k] = d_eta_ps[i];
    k++;
  }
  for (int i = 0; i < s_size; i++) {
    gradients[k] = d_eta_ss[i];
    k++;
  }
  return var(new precomputed_gradients_vari(
      tgt, 7 + theta_r_size + theta_s_size + p_size + p_size + s_size + s_size,
      varis, gradients));
}
*/
#endif
}  // namespace math
}  // namespace stan

#endif
