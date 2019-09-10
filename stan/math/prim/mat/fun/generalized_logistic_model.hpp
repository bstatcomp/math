#ifndef STAN_MATH_PRIM_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP
#define STAN_MATH_PRIM_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <algorithm>

namespace stan {
namespace math {

inline double dbeta(const double x, const double a, const double b) {
  return log(x) * (a - 1) + log(1 - x) * (b - 1) - lbeta(a, b);
}

inline double generalized_logistic_model(
    const std::vector<int> &IDp, const std::vector<int> &IDs,
    const std::vector<int> &is_pbo, const vector_d &time, const vector_d &score,
    const int multiplicative_s, const int multiplicative_r,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &X_s,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &X_r,
    double tau, double beta, double beta_pbo, double k_el, double k_eq,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &theta_r,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &theta_s,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &eta_pr,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &eta_sr,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &eta_ps,
    const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &eta_ss,
    const double base_s, const double base_r) {
  double tgt = 0;
  for (int i = 0; i < IDp.size(); i++) {
    int IDp_i = IDp[i] - 1;
    int IDs_i = IDs[i] - 1;
    // compute function
    double cov_s = base_s + eta_ps(IDp_i) + eta_ss(IDs_i);
    double cov_r = base_r + eta_pr(IDp_i) + eta_sr(IDs_i);

    if (theta_s.size() > 0)
      cov_s = cov_s + (X_s.row(i) * theta_s.col(0))(0, 0);
    if (theta_r.size() > 0) {
      cov_r = cov_r + (X_r.row(i) * theta_r.col(0))(0, 0);
    }
    if (multiplicative_s == 1) {
      cov_s = exp(cov_s);
    }
    if (multiplicative_r == 1) {
      cov_r = exp(cov_r);
    }

    const double S0 = 1 / (1 + exp(-cov_s));
    const double d_k_eq_el = k_eq - k_el;
    const double k_eq_n = k_eq / (d_k_eq_el);
    const double exp_k_el_eq_t = (exp(-k_el * time[i]) - exp(-k_eq * time[i]));
    const double pbo_eff = beta_pbo * k_eq_n * exp_k_el_eq_t;
    const double is_pbo_i = is_pbo[IDs_i];
    const double inv_beta = 1.0 / beta;
    const double S0_beta_pow = std::pow(S0, beta);
    const double beta_cov_t_prod = -beta * cov_r * time[i];
    const double exp_beta_cov_t_prod = exp(beta_cov_t_prod);
    const double alpha = std::pow(
        S0_beta_pow + (1 - S0_beta_pow) * exp_beta_cov_t_prod, inv_beta);
    double muS = S0 / alpha - is_pbo_i * pbo_eff;
    tgt += dbeta(score[i], muS * tau, (1 - muS) * tau);
  }
  return tgt;
}
}  // namespace math
}  // namespace stan

#endif