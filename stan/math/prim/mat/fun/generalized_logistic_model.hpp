#ifndef STAN_MATH_PRIM_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP
#define STAN_MATH_PRIM_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <algorithm>

namespace stan {
  namespace math {
	  
	inline double dbeta(double x, double a, double b) {

	return log(x) * (a - 1) + log(1 - x) * (b - 1) - lbeta(a, b);

	}  
    
    double generalized_logistic_model(const std::vector<int>& IDp, const std::vector<int>& IDs, const std::vector<int>& is_pbo, const vector_d& time, const vector_d& score, const int &multiplicative_s, const int &multiplicative_r, 
				 const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_s, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_r,
				 double &tau, double &beta, double &beta_pbo, double &k_el, double &k_eq,
				 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> theta_r, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> theta_s,
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eta_pr, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eta_sr,
                 Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eta_ps, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> eta_ss,
				 double &base_s, double &base_r
				 ) {
        int N = IDp.size();
        double tgt = 0;
        double cov_s = 0;
		double cov_r = 0;
		std::vector<double> muS(N);
	
		for(int i = 0; i < N; i++)
		{
			cov_s = base_s + eta_ps(IDp[i]-1) + eta_ss(IDs[i]-1);
			cov_r = base_r + eta_pr(IDp[i]-1) + eta_sr(IDs[i]-1);
			if (!theta_s.rows()==0) cov_s = cov_s + (X_s.row(i) * theta_s)(0,0);
			if (!theta_r.rows()==0) cov_r = cov_r + (X_r.row(i) * theta_r)(0,0);
			if (multiplicative_s == 1) cov_s = exp(cov_s);
			if (multiplicative_r == 1) cov_s = exp(cov_r);
			double S0 = 1 / (1 + exp(-cov_s));
			double pbo_eff = beta_pbo * (k_eq / (k_eq - k_el)) * (exp(-k_el * time[i]) - exp(-k_eq * time[i]));
			muS[i] = S0 / pow((pow(S0, beta) + (1 - pow(S0, beta))* exp(-beta * cov_r * time[i])), 1.0 / beta) - is_pbo[IDs[i]-1] * pbo_eff;

		}
		for (int i = 0; i < N; i++) {
			tgt += dbeta(score[i], muS[i] * tau, (1 - muS[i])*tau);
		}

		return tgt;
    }
  }
}

#endif