#ifndef STAN_MATH_REV_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP
#define STAN_MATH_REV_MAT_FUN_GENERALIZED_LOGISTIC_MODEL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <algorithm>

namespace stan {
  namespace math {
    
    stan::math::var model2_cpu(const std::vector<int>& IDp, const std::vector<int>& IDs, const vector_d& time, const vector_d& S, const vector_d& APOE4, const std::vector<int>& AGE,
                          const std::vector<int>& SEX, const std::vector<int>& pbo_flag, const std::vector<int>& COMED, stan::math::var &theta_S0v, stan::math::var &theta_rv,
                          stan::math::var &tauv, stan::math::var &theta_AGEv, stan::math::var &theta_APOE4_rv, stan::math::var &theta_APOE4_bv, stan::math::var &theta_COMEDv,
                          stan::math::var &betav, stan::math::var &theta_SEXv, stan::math::var &beta_batemanv, stan::math::var &kelv, stan::math::var &keqv, 
                          Eigen::Matrix<stan::math::var, -1, -1> eta_pb, Eigen::Matrix<stan::math::var, -1, -1> eta_pr,
                          Eigen::Matrix<stan::math::var, -1, -1> eta_sb, Eigen::Matrix<stan::math::var, -1, -1> eta_sr) {
      int N= IDp.size();
      int pb = eta_pb.size();
      int sb = eta_sb.size();
      
      double theta_S0 = theta_S0v.val();
      double theta_r = theta_rv.val();
      double tau = tauv.val();
      double theta_AGE = theta_AGEv.val();
      double theta_APOE4_r = theta_APOE4_rv.val();
      double theta_APOE4_b = theta_APOE4_bv.val();
      double theta_COMED = theta_COMEDv.val();
      double beta = betav.val();
      double theta_SEX = theta_SEXv.val();
      double beta_bateman = beta_batemanv.val();
      double kel = kelv.val();
      double keq = keqv.val();
      
      double d_tau = 0.0;
      double d_beta = 0.0;
      double d_bb = 0.0;
      double d_kel = 0.0;
      double d_keq = 0.0;

      double d_theta_r = 0.0;
      double d_theta_S0 = 0.0;
      double d_theta_AGE = 0;
      double d_theta_SEX = 0;
      double d_theta_APOE4_b = 0;
      double d_theta_APOE4_r = 0;
      double d_theta_COMED = 0;
      
      double* d_eta_pr = NULL;
      double* d_eta_pb = NULL;
      double* d_eta_sb = NULL;
      double* d_eta_sr = NULL;
      d_eta_pr = (double *)calloc(pb, sizeof(double));
      d_eta_pb = (double *)calloc(pb, sizeof(double));
      d_eta_sb = (double *)calloc(sb, sizeof(double));
      d_eta_sr = (double *)calloc(sb, sizeof(double));
      
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
          double tmp_exp1 = exp(eta_pb(IDp[i]-1).val() + eta_sb(IDs[i]-1).val());
          double logsi1 = log(1 - Si);
          double logsi = log(Si);
          double r = theta_r*(1 + theta_AGE*(AGEi - 75))*(1 + theta_APOE4_r*(APOE4i - 0.72))*(1 + theta_COMED*COMEDi) + eta_pr(IDp[i]-1).val() + eta_sr(IDs[i]-1).val();
          double S0 = theta_S0* (1 + theta_SEX*SEXi) * (1 + theta_APOE4_b * (APOE4i - 0.72))*tmp_exp1;
          double tmp_exp = exp(-(5 * beta + 5)*r*timei);
          double tmp_pow = std::pow(S0, (5 * beta + 5));
          double pbo = exp(beta_bateman - 3.5)*(exp(-exp(kel + 0.46)*timei) - exp(-exp(keq + 1.88)*timei));
          double muS = S0 / std::pow(tmp_pow + ((1 - tmp_pow) * tmp_exp), (1 / (5 * beta + 5)))-flagi*pbo;
          tgt = tgt + (muS*(80 * tau + 80) - 1) * logsi + ((1 - muS)*(80 * tau + 80) - 1) * logsi1 - lbeta(muS*(80 * tau + 80), (1 - muS)*(80 * tau + 80));
          
          double digamma_tmp1 = digamma(80 * (1 + tau) * muS);
          double digamma_tmp2 = digamma(-80 * (1 + tau)* (-1 + muS));
          double cs_mus = -80 * (1 + tau) * (logsi1 - logsi - digamma_tmp2 + digamma_tmp1);

          //tau
          d_tau = d_tau + -80 * (-log(1 - Si) + muS * log(1 - Si) - muS * log(Si) - digamma(80 * (1 + tau))
          - (-1 + muS) * digamma(-80 * (-1 + muS) * (1 + tau)) +  muS * digamma(80 * muS * (1 + tau)));
        
          double cs_z = r * timei;
          double tmp_01 = exp(5 * (1 + beta) * cs_z) - 1;
          double tmp_02 = std::pow(S0, (5 + 5 * beta));
          double tmp_03 = -exp(-5 * (1 + beta) * cs_z);
          double tmp_04 = -1 + tmp_02;
          double tmp_nom1 = std::pow((tmp_02 + tmp_03 * tmp_04), (-1 / (5 + 5 * beta)));
          double tmp_nom = S0 * tmp_nom1 * (-5 * tmp_04 * (1 + beta) * cs_z - 5 * tmp_01 * tmp_02 * (1 + beta) * log(S0) + (1 + tmp_01 * tmp_02) * log(tmp_02 + tmp_03 * tmp_04));
          double tmp_den = 5 * (1 + tmp_01 * tmp_02) * (1 + beta) * (1 + beta);
          d_beta = d_beta + cs_mus * (tmp_nom / tmp_den);

          //beta_bateman, kel, keq
          double	ca = 3.50;
          double	cb = 0.46;
          double	cc = 1.88;
          d_bb = d_bb - cs_mus * flagi * exp(beta_bateman - ca) * (exp(-exp(cb + kel) * timei) - exp(-exp(cc + keq) * timei));
          d_kel = d_kel - cs_mus * flagi * -timei * exp(beta_bateman - ca + cb + kel - exp(cb + kel) * timei);
          d_keq = d_keq - cs_mus * flagi * +timei * exp(beta_bateman - ca + cc + keq - exp(cc + keq) * timei);
          
          //theta_r
          double cs_r = -tmp_04 * (muS+ flagi*pbo) / (1 + (-1 + exp((5 * beta + 5)*r*timei))*(tmp_02));
          cs_r = cs_mus * cs_r * timei;
          
          d_theta_r = d_theta_r + cs_r*(1 + theta_AGE*(AGEi - 75))*(1 + theta_APOE4_r*(APOE4i - 0.72))*(1 + theta_COMED*COMEDi);
          d_theta_AGE = d_theta_AGE + cs_r * (AGEi - 75)*(1 + theta_APOE4_r*(APOE4i - 0.72))*(1 + theta_COMED*COMEDi) * theta_r;
          d_theta_APOE4_r = d_theta_APOE4_r + cs_r * (1 + theta_AGE*(AGEi - 75))*(APOE4i - 0.72)*(1 + theta_COMED*COMEDi) * theta_r;
          d_theta_COMED = d_theta_COMED + cs_r * (1 + theta_AGE*(AGEi - 75))*(1 + theta_APOE4_r*(APOE4i - 0.72)) * COMEDi * theta_r;
          
          double cs_x = tmp_nom1 / (1 + tmp_01  * tmp_02);
          d_theta_S0 = d_theta_S0 + cs_mus * cs_x * (1 + theta_SEX*SEXi)*(1 + theta_APOE4_b*(APOE4i - 0.72))*tmp_exp1;
          d_theta_SEX=d_theta_SEX + cs_mus * cs_x *  theta_S0 * SEXi * (1 + theta_APOE4_b*(APOE4i - 0.72))*tmp_exp1;
          d_theta_APOE4_b =d_theta_APOE4_b + cs_mus * cs_x *  theta_S0 * (1 + theta_SEX*SEXi)*((APOE4i - 0.72))*tmp_exp1;

          d_eta_pr[IDp[i]-1] = d_eta_pr[IDp[i]-1] + cs_r;
          d_eta_pb[IDp[i]-1] = d_eta_pb[IDp[i]-1] + cs_mus * cs_x * S0;
          d_eta_sb[IDs[i]-1] = d_eta_sb[IDs[i]-1] + cs_mus * cs_x * S0;
          d_eta_sr[IDs[i]-1] = d_eta_sr[IDs[i]-1] + cs_r;

      }

      vari** varis  = ChainableStack::instance().memalloc_.alloc_array<vari*>(12+pb+pb+sb+sb);//4+2*pb+2*sb);
      varis[0] = tauv.vi_;
      varis[1] = betav.vi_;;
      varis[2] = beta_batemanv.vi_;
      varis[3] = kelv.vi_;
      varis[4] = keqv.vi_;
      varis[5] = theta_rv.vi_;
      varis[6] = theta_AGEv.vi_;
      varis[7] = theta_APOE4_rv.vi_;
      varis[8] = theta_COMEDv.vi_;
      varis[9] = theta_S0v.vi_;
      varis[10] = theta_SEXv.vi_;
      varis[11] = theta_APOE4_bv.vi_;
      
      int k=12;
      for(int i=0;i<eta_pr.size();i++){
        varis[k]=eta_pr(i).vi_;
        k++;
      }
      for(int i=0;i<eta_pb.size();i++){
        varis[k]=eta_pb(i).vi_;
        k++;
      }
      for(int i=0;i<eta_sb.size();i++){
        varis[k]=eta_sb(i).vi_;
        k++;
      }
      for(int i=0;i<eta_sr.size();i++){
        varis[k]=eta_sr(i).vi_;
        k++;
      }
      double* gradients = ChainableStack::instance().memalloc_.alloc_array<double>(12+pb+pb+sb+sb);//4+2*pb+2*sb);
      gradients[0] = d_tau; //theta_S0
      gradients[1] = d_beta; //theta_r
      gradients[2] = d_bb; //tau
      gradients[3] = d_kel; //tau
      gradients[4] = d_keq; //tau
      gradients[5] = d_theta_r; //tau
      gradients[6] = d_theta_AGE; //tau
      gradients[7] = d_theta_APOE4_r; //tau
      gradients[8] = d_theta_COMED; //tau
      gradients[9] = d_theta_S0; //tau
      gradients[10] = d_theta_SEX; //tau
      gradients[11] = d_theta_APOE4_b; //tau
      
      k=12;
      for(int i=0;i<pb;i++){
        gradients[k]=d_eta_pr[i];
        k++;
      }
      for(int i=0;i<pb;i++){
        gradients[k]=d_eta_pb[i];
        k++;
      }
      for(int i=0;i<sb;i++){
        gradients[k]=d_eta_sb[i];
        k++;
      }
      for(int i=0;i<sb;i++){
        gradients[k]=d_eta_sr[i];
        k++;
      }      
      free(d_eta_sb);
      free(d_eta_sr);
      free(d_eta_pb);
      free(d_eta_pr);
      return var(new precomputed_gradients_vari(tgt, 12+pb+pb+sb+sb,  varis, gradients));
    }
  }
}

#endif
