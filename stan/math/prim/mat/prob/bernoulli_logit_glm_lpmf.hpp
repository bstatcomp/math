#ifndef STAN_MATH_PRIM_MAT_PROB_BERNOULLI_LOGIT_GLM_LPMF_HPP
#define STAN_MATH_PRIM_MAT_PROB_BERNOULLI_LOGIT_GLM_LPMF_HPP

#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/broadcast_array.hpp>
#include <stan/math/prim/scal/meta/operands_and_partials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_consistent_size.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/arr/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/fun/as_array_or_scalar.hpp>
#include <stan/math/prim/scal/fun/as_scalar.hpp>
#include <stan/math/prim/mat/fun/as_scalar.hpp>
#include <stan/math/prim/arr/fun/as_scalar.hpp>
#include <stan/math/prim/mat/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/scal/fun/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/scal/meta/scalar_seq_view.hpp>
#include <stan/math/prim/scal/fun/size_zero.hpp>
#include <boost/random/variate_generator.hpp>
#ifdef STAN_OPENCL
#include <stan/math/prim/mat/fun/value_of.hpp>
#include <stan/math/prim/arr/fun/value_of.hpp>
#endif

#include <stan/math/opencl/kernels/bernoulli_logit_glm_lpmf.hpp>

#include <cmath>

namespace stan {
namespace math {

/**
 * Returns the log PMF of the Generalized Linear Model (GLM)
 * with Bernoulli distribution and logit link function.
 * The idea is that bernoulli_logit_glm_lpmf(y, x, alpha, beta) should
 * compute a more efficient version of bernoulli_logit_lpmf(y, alpha + x * beta)
 * by using analytically simplified gradients.
 * If containers are supplied, returns the log sum of the probabilities.
 * @tparam T_y type of binary vector of dependent variables (labels);
 * this can also be a single binary value;
 * @tparam T_x type of the matrix of independent variables (features); this
 * should be an Eigen::Matrix type whose number of rows should match the
 * length of y and whose number of columns should match the length of beta
 * @tparam T_alpha type of the intercept(s);
 * this can be a vector (of the same length as y) of intercepts or a single
 * value (for models with constant intercept);
 * @tparam T_beta type of the weight vector;
 * this can also be a single value;
 * @param y binary vector parameter
 * @param x design matrix
 * @param alpha intercept (in log odds)
 * @param beta weight vector
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if x, beta or alpha is infinite.
 * @throw std::domain_error if y is not binary.
 * @throw std::invalid_argument if container sizes mismatch.
 */

template <bool propto, typename T_y, typename T_x, typename T_alpha,
          typename T_beta>
typename return_type<T_x, T_alpha, T_beta>::type bernoulli_logit_glm_lpmf(
    const T_y &y, const T_x &x, const T_alpha &alpha, const T_beta &beta) {
  static const char *function = "bernoulli_logit_glm_lpmf";
  typedef typename stan::partials_return_type<T_y, T_x, T_alpha, T_beta>::type
      T_partials_return;
  typedef typename std::conditional<
      is_vector<T_y>::value,
      Eigen::Matrix<typename stan::partials_return_type<T_y>::type, -1, 1>,
      typename stan::partials_return_type<T_y>::type>::type T_y_val;

  using Eigen::Dynamic;
  using Eigen::Matrix;
  using std::exp;

  if (size_zero(y, x, beta))
    return 0.0;

  T_partials_return logp(0.0);

  const size_t N = x.rows();
  const size_t M = x.cols();

#ifndef STAN_OPENCL
  check_bounded(function, "Vector of dependent variables", y, 0, 1);
#endif
  check_consistent_size(function, "Vector of dependent variables", y, N);
  check_consistent_size(function, "Weight vector", beta, M);
  if (is_vector<T_alpha>::value)
    check_consistent_sizes(function, "Vector of intercepts", alpha,
                           "Vector of dependent variables", y);

  if (!include_summand<propto, T_x, T_alpha, T_beta>::value)
    return 0.0;

  const auto &x_val = value_of_rec(x);
#ifdef STAN_OPENCL
  const auto &y_val = value_of(y);
#else
  const auto &y_val = value_of_rec(y);
#endif
  const auto &beta_val = value_of_rec(beta);
  const auto &alpha_val = value_of_rec(alpha);

  const auto &y_val_vec = as_column_vector_or_scalar(y_val);
  const auto &beta_val_vec = as_column_vector_or_scalar(beta_val);
  const auto &alpha_val_vec = as_column_vector_or_scalar(alpha_val);

#ifdef STAN_OPENCL
  const int local_size = opencl_kernels::bernoulli_logit_glm.make_functor.get_opts().at("LOCAL_SIZE_");
  const int wgs = (N+local_size-1)/local_size;

  const matrix_cl y_cl = matrix_cl::constant(y_val_vec);
  const matrix_cl x_cl = matrix_cl::constant(x_val);
  const matrix_cl beta_cl(beta_val_vec);
  const matrix_cl alpha_cl(alpha_val_vec);

  matrix_cl logp_cl(wgs, 1);
  const bool need_theta_derivative = !is_constant_struct<T_beta>::value || !is_constant_struct<T_x>::value || !is_constant_struct<T_alpha>::value;
  matrix_cl theta_derivative_cl(need_theta_derivative ? N : 0, 1);
  const bool need_theta_derivative_sum = need_theta_derivative && !is_vector<T_alpha>::value;
  matrix_cl theta_derivative_sum_cl(need_theta_derivative_sum ? wgs : 0, 1);

  try {
    opencl_kernels::bernoulli_logit_glm(cl::NDRange(local_size * wgs), cl::NDRange(local_size),
                                        y_cl.buffer(), x_cl.buffer(), alpha_cl.buffer(), beta_cl.buffer(),
                                        logp_cl.buffer(), theta_derivative_cl.buffer(), theta_derivative_sum_cl.buffer(),
                                        N, M, length(alpha) != 1, need_theta_derivative, need_theta_derivative_sum);
  }
  catch (const cl::Error& e) {
    check_opencl_error(function, e);
  }

  Eigen::VectorXd logp_partial_sum(wgs);
  copy(logp_partial_sum, logp_cl);
  logp += sum(logp_partial_sum);
#else
  T_y_val signs = 2 * as_array_or_scalar(y_val_vec) - 1;

  Eigen::Array<T_partials_return, Dynamic, 1> ytheta = x_val * beta_val_vec;
  ytheta = as_array_or_scalar(signs)
           * (ytheta + as_array_or_scalar(alpha_val_vec));

  // Compute the log-density and handle extreme values gracefully
  // using Taylor approximations.
  // And compute the derivatives wrt theta.
  static const double cutoff = 20.0;
  Eigen::Array<T_partials_return, Dynamic, 1> exp_m_ytheta = exp(-ytheta);
  logp += (ytheta > cutoff)
              .select(-exp_m_ytheta,
                      (ytheta < -cutoff).select(ytheta, -log1p(exp_m_ytheta)))
              .sum();
#endif
  if (!std::isfinite(logp)) {
#ifdef STAN_OPENCL
    check_bounded(function, "Vector of dependent variables", y, 0, 1);
#endif
    check_finite(function, "Weight vector", beta);
    check_finite(function, "Intercept", alpha);
    check_finite(function, "Matrix of independent variables", x);
  }

  // Compute the necessary derivatives.
  operands_and_partials<T_x, T_alpha, T_beta> ops_partials(x, alpha, beta);
  if (!is_constant_struct<T_beta>::value || !is_constant_struct<T_x>::value
      || !is_constant_struct<T_alpha>::value) {
#ifdef STAN_OPENCL
    Matrix<T_partials_return, Dynamic, 1> theta_derivative(N);
    if(!is_constant_struct<T_x>::value || (!is_constant_struct<T_alpha>::value && is_vector<T_alpha>::value)) {
      copy(theta_derivative, theta_derivative_cl);
    }
#else
    Matrix<T_partials_return, Dynamic, 1> theta_derivative
        = (ytheta > cutoff)
              .select(-exp_m_ytheta,
                      (ytheta < -cutoff)
                          .select(as_array_or_scalar(signs),
                                  as_array_or_scalar(signs) * exp_m_ytheta
                                      / (exp_m_ytheta + 1)));
#endif
    if (!is_constant_struct<T_beta>::value) {
#ifdef STAN_OPENCL
      const matrix_cl theta_derivative_transpose_cl(*const_cast<cl::Buffer*>(&theta_derivative_cl.buffer()), 1, theta_derivative_cl.rows()); //transposition of a vector can be done without copying
      const matrix_cl beta_derivative_cl = theta_derivative_transpose_cl * x_cl;
      Eigen::RowVectorXd beta_derivative(M);
      copy(beta_derivative, beta_derivative_cl);
      ops_partials.edge3_.partials_ = std::move(beta_derivative);
#else
      ops_partials.edge3_.partials_ = x_val.transpose() * theta_derivative;
#endif
    }
    if (!is_constant_struct<T_x>::value) {
      ops_partials.edge1_.partials_
          = (beta_val_vec * theta_derivative.transpose()).transpose();
    }
    if (!is_constant_struct<T_alpha>::value) {
      if (is_vector<T_alpha>::value) {
        ops_partials.edge2_.partials_ = std::move(theta_derivative);
      }
      else {
#ifdef STAN_OPENCL
        Matrix<T_partials_return, Dynamic, 1> theta_derivative_partial_sum(wgs);
        copy(theta_derivative_partial_sum, theta_derivative_sum_cl);
        ops_partials.edge2_.partials_[0] = sum(theta_derivative_partial_sum);
#else
        ops_partials.edge2_.partials_[0] = theta_derivative.sum();
#endif
      }
    }
  }
  return ops_partials.build(logp);
}

template <typename T_y, typename T_x, typename T_alpha, typename T_beta>
inline typename return_type<T_x, T_beta, T_alpha>::type
bernoulli_logit_glm_lpmf(const T_y &y, const T_x &x, const T_alpha &alpha,
                         const T_beta &beta) {
  return bernoulli_logit_glm_lpmf<false>(y, x, alpha, beta);
}
}  // namespace math
}  // namespace stan
#endif
