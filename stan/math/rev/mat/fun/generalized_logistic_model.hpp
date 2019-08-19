#ifndef STAN_MATH_REV_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP
#define STAN_MATH_REV_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <algorithm>

namespace stan {
  namespace math {
      
    
    /*stan::math::var generalized_logistic_model(const std::vector<int>& IDp, const std::vector<int>& IDs, const vector_d& time, const vector_d& S, const vector_d& APOE4, const std::vector<int>& AGE,
                          const std::vector<int>& SEX, const std::vector<int>& pbo_flag, const std::vector<int>& COMED, stan::math::var &theta_S0v, stan::math::var &theta_rv,
                          stan::math::var &tauv, stan::math::var &theta_AGEv, stan::math::var &theta_APOE4_rv, stan::math::var &theta_APOE4_bv, stan::math::var &theta_COMEDv,
                          stan::math::var &betav, stan::math::var &theta_SEXv, stan::math::var &beta_batemanv, stan::math::var &kelv, stan::math::var &keqv, 
                          Eigen::Matrix<stan::math::var, -1, -1> eta_pb, Eigen::Matrix<stan::math::var, -1, -1> eta_pr,
                          Eigen::Matrix<stan::math::var, -1, -1> eta_sb, Eigen::Matrix<stan::math::var, -1, -1> eta_sr) {
      */
    stan::math::var generalized_logistic_model(const std::vector<int>& IDp, const std::vector<int>& IDs, const std::vector<int>& is_pbo, const vector_d& time, const vector_d& score, const int &multiplicative_s, const int &multiplicative_r, 
                 const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_s, const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& X_r,
                 stan::math::var &tauv, stan::math::var &betav, stan::math::var &beta_pbov, stan::math::var &k_elv, stan::math::var &k_eqv,
                 Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> theta_r, Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> theta_s,
                 Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> eta_pr, Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> eta_sr,
                 Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> eta_ps, Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> eta_ss,
                 stan::math::var &base_sv, stan::math::var &base_rv
                 ) {
        int N = IDp.size();
      
        double tau = tauv.val();
        double beta = betav.val();
        double beta_pbo = beta_pbov.val();
        double k_el = k_elv.val();
        double k_eq = k_eqv.val();
        double base_s = base_sv.val();
        double base_r = base_rv.val();
      
        std::vector<double> muS(N);
        double d_tau = 0;
        double d_beta_pbo = 0;
        double  d_k_eq = 0;
        double  d_k_el = 0;
        double  d_beta = 0;
        double  d_base_s = 0;
        double  d_base_r = 0;
        std::vector<double> d_eta_pr(*std::max_element(IDp.begin(), IDp.end())+1);
        std::vector<double> d_eta_sr(*std::max_element(IDs.begin(), IDs.end())+1);
        std::vector<double> d_eta_ps(*std::max_element(IDp.begin(), IDp.end())+1);
        std::vector<double> d_eta_ss(*std::max_element(IDs.begin(), IDs.end())+1);
        std::vector<double> d_theta_s(2);
        std::vector<double> d_theta_r(2);
        
        int r_size=d_eta_pr.size();
        int s_size=d_eta_ps.size();
        
        //Main loop
        double cov_s = 0;
        double cov_r = 0;
        double tgt=0;
        for(int i = 0; i < N; i++)
		{
            //compute function
			cov_s = base_s + eta_ps(IDp[i]-1).val() + eta_ss(IDs[i]-1).val();
			cov_r = base_r + eta_pr(IDp[i]-1).val() + eta_sr(IDs[i]-1).val();
			if (!theta_s.rows()==0) cov_s = cov_s + (X_s.row(i) * theta_s)(0,0).val();
			if (!theta_r.rows()==0) cov_r = cov_r + (X_r.row(i) * theta_r)(0,0).val();
			if (multiplicative_s == 1) cov_s = exp(cov_s);
			if (multiplicative_r == 1) cov_s = exp(cov_r);
			double S0 = 1 / (1 + exp(-cov_s));
			double pbo_eff = beta_pbo * (k_eq / (k_eq - k_el)) * (exp(-k_el * time[i]) - exp(-k_eq * time[i]));
			muS[i] = S0 / std::pow((std::pow(S0, beta) + (1 - std::pow(S0, beta))* exp(-beta * cov_r * time[i])), 1.0 / beta) - is_pbo[IDs[i]-1] * pbo_eff;

            //compute gradients
            double d_x_d_mu = tau * log(score[i]) - tau * log(1 - score[i]) - digamma(muS[i] * tau) * tau + digamma(tau - muS[i] * tau) * tau;
            d_tau = d_tau + muS[i] * log(score[i]) + log(1 - score[i]) - muS[i] * log(1 - score[i]) - digamma(muS[i] * tau) * muS[i] - digamma(tau - muS[i] * tau) * (1 - muS[i]) + digamma(tau);

            d_beta_pbo = d_beta_pbo + d_x_d_mu * (-is_pbo[IDs[i]-1])*(k_eq / (k_eq - k_el)) * (exp(-k_el* time[i]) - exp(-k_eq * time[i]));
            d_k_eq = d_k_eq + d_x_d_mu * (-is_pbo[IDs[i]-1]) * beta_pbo * (((-k_el) / ((k_eq - k_el)*(k_eq - k_el))) * (exp(-k_el*time[i]) - exp(-k_eq * time[i])) + (k_eq / (k_eq - k_el)) * exp(-k_eq * time[i]) * time[i]);
		
            d_k_el = d_k_el + d_x_d_mu * (-is_pbo[IDs[i]-1]) * beta_pbo * ((k_eq / ((k_eq - k_el)*(k_eq - k_el))) * (exp(-k_el*time[i]) - exp(-k_eq * time[i])) - (k_eq / (k_eq - k_el)) * exp(-k_el * time[i]) * time[i]);

            double alpha = std::pow(std::pow(S0, beta) + (1 - std::pow(S0, beta))*exp(-beta * cov_r *time[i]), 1.0 / beta);
            double tmp_s = d_x_d_mu * ((alpha - S0 * (1 / beta)*alpha*std::pow(alpha, -beta)*(beta*std::pow(S0,beta-1)-beta*exp(-beta*cov_r*time[i])*std::pow(S0,beta-1)))/(alpha*alpha))* (exp(-cov_s) / ((1 + exp(-cov_s))*(1 + exp(-cov_s))));
            double tmp_r = d_x_d_mu * S0 * (-((1 / beta)*alpha*std::pow(alpha, -beta) * (-exp(-beta*cov_r*time[i])*beta*time[i] + std::pow(S0, beta)*exp(-beta * cov_r * time[i])* beta * time[i])) / (alpha * alpha));
            if (multiplicative_s == 1) tmp_s = tmp_s * cov_s;
            if (multiplicative_r == 1) tmp_r = tmp_r * cov_r;

            d_base_s += tmp_s;
            d_theta_s[0] += tmp_s * X_s(i,0);
            d_theta_s[1] += tmp_s * X_s(i,1);
            d_eta_ps[IDp[i]-1] = d_eta_ps[IDp[i]-1] + tmp_s;
            d_eta_ss[IDs[i]-1] = d_eta_ss[IDs[i]-1] + tmp_s;
		
            d_base_r += tmp_r;
            d_theta_r[0] += tmp_r * X_r(i,0);
            d_theta_r[1] += tmp_r * X_r(i,1);
            d_eta_pr[IDp[i]-1] = d_eta_pr[IDp[i]-1] + tmp_r;
            d_eta_sr[IDs[i]-1] = d_eta_sr[IDs[i]-1] + tmp_r;

            d_beta += d_x_d_mu * (-S0 * std::pow(alpha, -2)) * alpha * ((log(std::pow(alpha, beta)) * (-std::pow(beta, -2))) +
                    ((1 / beta) * (std::pow(S0, beta) * log(S0) - exp(-beta * cov_r * time[i]) * cov_r * time[i] -
                    std::pow(S0, beta) * log(S0) * exp(-beta * cov_r * time[i]) +
                    std::pow(S0, beta) * exp(-beta * cov_r * time[i]) * cov_r * time[i])) / std::pow(alpha, beta));

           
        
        }
        for (int i = 0; i < N; i++) {
			tgt += dbeta(score[i], muS[i] * tau, (1 - muS[i])*tau);
		}
 
      //stack it up
      vari** varis  = ChainableStack::instance().memalloc_.alloc_array<vari*>(7+4+r_size+r_size+s_size+s_size);
      varis[0] = tauv.vi_;
      varis[1] = betav.vi_;;
      varis[2] = beta_pbov.vi_;
      varis[3] = k_elv.vi_;
      varis[4] = k_eqv.vi_;
      varis[5] = base_sv.vi_;
      varis[6] = base_rv.vi_;
      
      int k=7;
      for(int i=0;i<theta_r.size();i++){
        varis[k]=theta_r(i).vi_;
        k++;
      }
      for(int i=0;i<theta_s.size();i++){
        varis[k]=theta_s(i).vi_;
        k++;
      }
      for(int i=0;i<eta_pr.size();i++){
        varis[k]=eta_pr(i).vi_;
        k++;
      }
      for(int i=0;i<eta_sr.size();i++){
        varis[k]=eta_sr(i).vi_;
        k++;
      }
      for(int i=0;i<eta_ps.size();i++){
        varis[k]=eta_ps(i).vi_;
        k++;
      }
      for(int i=0;i<eta_ss.size();i++){
        varis[k]=eta_ss(i).vi_;
        k++;
      }
      double* gradients = ChainableStack::instance().memalloc_.alloc_array<double>(7+4+r_size+r_size+s_size+s_size);
      gradients[0] = d_tau;
      gradients[1] = d_beta; 
      gradients[2] = d_beta_pbo;
      gradients[3] = d_k_el; 
      gradients[4] = d_k_eq; 
      gradients[5] = d_base_r;
      gradients[6] = d_base_s; 
      
      k=7;
      for(int i=0;i<2;i++){
        gradients[k]=d_theta_r[i];
        k++;
      }
      for(int i=0;i<2;i++){
        gradients[k]=d_theta_s[i];
        k++;
      }
      for(int i=0;i<r_size;i++){
        gradients[k]=d_eta_pr[i];
        k++;
      }
      for(int i=0;i<r_size;i++){
        gradients[k]=d_eta_sr[i];
        k++;
      }
      for(int i=0;i<s_size;i++){
        gradients[k]=d_eta_ps[i];
        k++;
      }
      for(int i=0;i<s_size;i++){
        gradients[k]=d_eta_ss[i];
        k++;
      }      
      return var(new precomputed_gradients_vari(tgt, 7+4+r_size+r_size+s_size+s_size,  varis, gradients));
    }
  }
}

#endif
