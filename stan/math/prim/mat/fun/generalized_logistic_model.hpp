#ifndef STAN_MATH_PRIM_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP
#define STAN_MATH_PRIM_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/cholesky_decompose.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <algorithm>

namespace stan {
  namespace math {
    
    double generalized_logistic_model(const std::vector<int>& IDp, const std::vector<int>& IDs, const vector_d& time, const vector_d& S, const vector_d& APOE4, const std::vector<int>& AGE,
                 const std::vector<int>& SEX, const std::vector<int>& pbo_flag, const std::vector<int>& COMED, const double &theta_S0, double &theta_r, double &tau,
                 double &theta_AGE, double &theta_APOE4_r, double &theta_APOE4_b, double &theta_COMED, double &beta, double &theta_SEX, double &beta_bateman,
                 double &kel, double &keq, 
                 Eigen::Matrix<double, -1, -1> eta_pb, Eigen::Matrix<double, -1, -1> eta_pr,
                 Eigen::Matrix<double, -1, -1> eta_sb, Eigen::Matrix<double, -1, -1> eta_sr) {
        int N= IDp.size();
        double tgt = 0;
        for (int i = 0; i < N; i++) {
          //preload all data
          double APOE4i = APOE4[i];
          int AGEi = AGE[i];
          int SEXi = SEX[i];
          double timei = time[i];
          double Si = S[i];
          int flagi = pbo_flag[i];
          int COMEDi = COMED[i];
          //-----------compute function
          double tmp_exp1 = exp(eta_pb(IDp[i]-1) + eta_sb(IDs[i]-1));
          double logsi1 = log(1 - Si);
          double logsi = log(Si);
          double r = theta_r*(1 + theta_AGE*(AGEi - 75))*(1 + theta_APOE4_r*(APOE4i - 0.72))*(1 + theta_COMED*COMEDi) + eta_pr(IDp[i]-1) + eta_sr(IDs[i]-1);
          double S0 = theta_S0* (1 + theta_SEX*SEXi) * (1 + theta_APOE4_b * (APOE4i - 0.72))*tmp_exp1;
          double tmp_exp = exp(-(5 * beta + 5)*r*timei);
          double tmp_pow = std::pow(S0, (5 * beta + 5));
          double pbo = exp(beta_bateman - 3.5)*(exp(-exp(kel + 0.46)*timei) - exp(-exp(keq + 1.88)*timei));
          double muS = S0 / std::pow(tmp_pow + ((1 - tmp_pow) * tmp_exp), (1 / (5 * beta + 5)))-flagi*pbo;
          tgt = tgt + (muS*(80 * tau + 80) - 1) * logsi + ((1 - muS)*(80 * tau + 80) - 1) * logsi1 - lbeta(muS*(80 * tau + 80), (1 - muS)*(80 * tau + 80));
        }      
        return tgt;
    }
  }
}

#endif