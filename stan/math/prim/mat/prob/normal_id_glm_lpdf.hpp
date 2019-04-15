#ifndef STAN_MATH_PRIM_MAT_PROB_NORMAL_ID_GLM_LPDF_HPP
#define STAN_MATH_PRIM_MAT_PROB_NORMAL_ID_GLM_LPDF_HPP

#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/operands_and_partials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/scal/meta/scalar_seq_view.hpp>
#include <stan/math/prim/scal/fun/sum.hpp>
#include <stan/math/prim/scal/meta/as_array_or_scalar.hpp>
#include <stan/math/prim/scal/meta/as_scalar.hpp>
#include <stan/math/prim/mat/meta/as_scalar.hpp>
#include <stan/math/prim/arr/meta/as_scalar.hpp>
#include <stan/math/prim/mat/meta/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/scal/meta/as_column_vector_or_scalar.hpp>
#include <stan/math/prim/arr/fun/value_of_rec.hpp>

#include <stan/math/opencl/kernels/normal_id_glm_lpdf.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/multiply.hpp>

#include <cmath>

namespace stan {
namespace math {

/**
 * Returns the log PDF of the Generalized Linear Model (GLM)
 * with Normal distribution and id link function.
 * If containers are supplied, returns the log sum of the probabilities.
 * The idea is that normal_id_glm_lpdf(y, x, alpha, beta, sigma) should
 * compute a more efficient version of normal_lpdf(y, alpha + x * beta, sigma)
 * by using analytically simplified gradients.
 * @tparam T_y type of vector of dependent variables (labels);
 * @tparam T_x type of the matrix of independent variables (features); this
 * should be an Eigen::Matrix type whose number of rows should match the
 * length of y and whose number of columns should match the length of beta
 * @tparam T_alpha type of the intercept(s);
 * this can be a vector (of the same length as y) of intercepts or a single
 * value (for models with constant intercept);
 * @tparam T_beta type of the weight vector;
 * this can also be a single value;
 * @tparam T_scale type of the (positive) scale(s);
 * this can be a vector (of the same length as y, for heteroskedasticity)
 * or a scalar.
 * @param y vector parameter
 * @param x design matrix
 * @param alpha intercept (in log odds)
 * @param beta weight vector
 * @param sigma (Sequence of) scale parameters for the normal
 * distribution.
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if x, beta or alpha is infinite.
 * @throw std::domain_error if the scale is not positive.
 * @throw std::invalid_argument if container sizes mismatch.
 */
template <bool propto, typename T_y, typename T_x, typename T_alpha,
          typename T_beta, typename T_scale>
typename return_type<T_y, T_x, T_alpha, T_beta, T_scale>::type
normal_id_glm_lpdf(const T_y &y, const T_x &x, const T_alpha &alpha,
                   const T_beta &beta, const T_scale &sigma) {
  static const char *function = "normal_id_glm_lpdf";
  typedef typename stan::partials_return_type<T_y, T_x, T_alpha, T_beta,
                                              T_scale>::type T_partials_return;
  typedef typename std::conditional<
      is_vector<T_scale>::value,
      Eigen::Array<typename stan::partials_return_type<T_scale>::type, -1, 1>,
      typename stan::partials_return_type<T_scale>::type>::type T_scale_val;

  using Eigen::Array;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using Eigen::VectorXd;
  using std::exp;

  if (!(stan::length(y) && stan::length(x) && stan::length(beta)
        && stan::length(sigma)))
    return 0.0;

  const size_t N = x.rows();
  const size_t M = x.cols();

  check_positive_finite(function, "Scale vector", sigma);
  check_consistent_size(function, "Vector of dependent variables", y, N);
  check_consistent_size(function, "Weight vector", beta, M);
  if (is_vector<T_scale>::value)
    check_consistent_sizes(function, "Vector of scale parameters", sigma,
                           "Vector of dependent variables", y);
  if (is_vector<T_alpha>::value)
    check_consistent_sizes(function, "Vector of intercepts", alpha,
                           "Vector of dependent variables", y);

  if (!include_summand<propto, T_y, T_x, T_alpha, T_beta, T_scale>::value)
    return 0.0;

  const auto &x_val = value_of_rec(x);
  const auto &beta_val = value_of_rec(beta);
  const auto &alpha_val = value_of_rec(alpha);
  const auto &sigma_val = value_of_rec(sigma);
  const auto &y_val = value_of_rec(y);

  const auto &beta_val_vec = as_column_vector_or_scalar(beta_val);
  const auto &alpha_val_vec = as_column_vector_or_scalar(alpha_val);
  const auto &sigma_val_vec = as_column_vector_or_scalar(sigma_val);
  const auto &y_val_vec = as_column_vector_or_scalar(y_val);

  T_scale_val inv_sigma = 1 / as_array_or_scalar(sigma_val_vec);
  Matrix<T_partials_return, Dynamic, 1> y_minus_mu_over_sigma_mat(N);
  auto y_minus_mu_over_sigma = y_minus_mu_over_sigma_mat.array();
#ifdef STAN_OPENCL
  const int local_size = opencl_kernels::normal_id_glm.make_functor.get_opts().at("LOCAL_SIZE_");
  const int wgs = (N+local_size-1)/local_size;

  const matrix_cl y_cl = matrix_cl::constant(y_val_vec);
  const matrix_cl x_cl = matrix_cl::constant(x_val);
  const matrix_cl beta_cl(beta_val_vec);
  const matrix_cl alpha_cl(alpha_val_vec);
  const matrix_cl sigma_cl(sigma_val_vec);

  const bool need_mu_derivative = !(is_constant_struct<T_y>::value && is_constant_struct<T_x>::value && is_constant_struct<T_beta>::value && is_constant_struct<T_alpha>::value);
  matrix_cl mu_derivative_cl(need_mu_derivative ? N : 0, 1);
  const bool need_mu_derivative_sum = !is_constant_struct<T_alpha>::value && !is_vector<T_alpha>::value;
  matrix_cl mu_derivative_sum_cl(need_mu_derivative_sum ? wgs : 0, 1);
  matrix_cl y_minus_mu_over_sigma_squared_sum_cl(wgs, 1);
  const bool need_sigma_derivative = !is_constant_struct<T_scale>::value && is_vector<T_scale>::value;
  matrix_cl sigma_derivative_cl(need_sigma_derivative ? N : 0, 1);
  const bool need_log_sigma_sum = include_summand<propto, T_scale>::value && is_vector<T_scale>::value;
  matrix_cl log_sigma_sum_cl(need_log_sigma_sum ? wgs : 0, 1);

  try {
    opencl_kernels::normal_id_glm(cl::NDRange(local_size * wgs), cl::NDRange(local_size),
                                  y_cl.buffer(), x_cl.buffer(), alpha_cl.buffer(), beta_cl.buffer(), sigma_cl.buffer(),
                                  mu_derivative_cl.buffer(), mu_derivative_sum_cl.buffer(), y_minus_mu_over_sigma_squared_sum_cl.buffer(), sigma_derivative_cl.buffer(), log_sigma_sum_cl.buffer(),
                                  N, M, length(alpha) != 1, length(sigma) != 1, need_mu_derivative, need_mu_derivative_sum, need_sigma_derivative, need_log_sigma_sum);
  }
  catch (const cl::Error& e) {
    check_opencl_error(function, e);
  }
  VectorXd y_minus_mu_over_sigma_squared_partial_sum(wgs);
  copy(y_minus_mu_over_sigma_squared_partial_sum, y_minus_mu_over_sigma_squared_sum_cl);
  double y_minus_mu_over_sigma_squared_sum = sum(y_minus_mu_over_sigma_squared_partial_sum);
#else
  y_minus_mu_over_sigma = x_val * beta_val_vec;
  y_minus_mu_over_sigma = (as_array_or_scalar(y_val_vec) - y_minus_mu_over_sigma
                           - as_array_or_scalar(alpha_val_vec))
                          * inv_sigma;
  double y_minus_mu_over_sigma_squared_sum;  // the most efficient way to
                                             // calculate this depends on
                                             // template parameters
#endif

  // Compute the derivatives.
  operands_and_partials<T_y, T_x, T_alpha, T_beta, T_scale> ops_partials(
      y, x, alpha, beta, sigma);
  if (!(is_constant_struct<T_y>::value && is_constant_struct<T_x>::value
        && is_constant_struct<T_beta>::value
        && is_constant_struct<T_alpha>::value)) {
#ifdef STAN_OPENCL
    Matrix<T_partials_return, Dynamic, 1> mu_derivative(N);
    if(!is_constant_struct<T_y>::value || !is_constant_struct<T_x>::value || (!is_constant_struct<T_alpha>::value && is_vector<T_alpha>::value)) {
      copy(mu_derivative, mu_derivative_cl);
    }
#else
    Matrix<T_partials_return, Dynamic, 1> mu_derivative
        = inv_sigma * y_minus_mu_over_sigma;
#endif
    if (!is_constant_struct<T_y>::value) {
      ops_partials.edge1_.partials_ = -mu_derivative;
    }
    if (!is_constant_struct<T_x>::value) {
      ops_partials.edge2_.partials_
          = (beta_val_vec * mu_derivative.transpose()).transpose();
    }
    if (!is_constant_struct<T_beta>::value) {
#ifdef STAN_OPENCL
      const matrix_cl mu_derivative_transpose_cl(*const_cast<cl::Buffer*>(&mu_derivative_cl.buffer()), 1, mu_derivative_cl.rows()); //transposition of a vector can be done without copying
      const matrix_cl beta_derivative_cl = mu_derivative_transpose_cl * x_cl;
      Eigen::RowVectorXd beta_derivative(M);
      copy(beta_derivative, beta_derivative_cl);
      ops_partials.edge4_.partials_ = std::move(beta_derivative);
#else
      ops_partials.edge4_.partials_ = mu_derivative.transpose() * x_val;
#endif
    }
    if (!is_constant_struct<T_alpha>::value) {
      if (is_vector<T_alpha>::value)
        ops_partials.edge3_.partials_ = mu_derivative;
      else {
#ifdef STAN_OPENCL
        VectorXd mu_derivative_partial_sum(wgs);
        copy(mu_derivative_partial_sum, mu_derivative_sum_cl);
        ops_partials.edge3_.partials_[0] = sum(mu_derivative_partial_sum);
#else
        ops_partials.edge3_.partials_[0] = mu_derivative.sum();
#endif
      }
    }
    if (!is_constant_struct<T_scale>::value) {
      if (is_vector<T_scale>::value) {
#ifdef STAN_OPENCL
        VectorXd sigma_derivative(N);
        copy(sigma_derivative, sigma_derivative_cl);
        ops_partials.edge5_.partials_ = std::move(sigma_derivative);
#else
        Array<T_partials_return, Dynamic, 1> y_minus_mu_over_sigma_squared
            = y_minus_mu_over_sigma * y_minus_mu_over_sigma;
        y_minus_mu_over_sigma_squared_sum = y_minus_mu_over_sigma_squared.sum();
        ops_partials.edge5_.partials_
            = (y_minus_mu_over_sigma_squared - 1) * inv_sigma;
#endif
      } else {
#ifndef STAN_OPENCL
        y_minus_mu_over_sigma_squared_sum
            = (y_minus_mu_over_sigma * y_minus_mu_over_sigma).sum();
#endif
        ops_partials.edge5_.partials_[0]
            = (y_minus_mu_over_sigma_squared_sum - N) * as_scalar(inv_sigma);
      }
    }
  }
#ifndef STAN_OPENCL
  else {
    y_minus_mu_over_sigma_squared_sum
        = (y_minus_mu_over_sigma * y_minus_mu_over_sigma).sum();
  }
#endif

  if (!std::isfinite(
          y_minus_mu_over_sigma_squared_sum)) {  // only do potentially
                                                 // expensive checks if they are
                                                 // really needed
    check_finite(function, "Vector of dependent variables", y);
    check_finite(function, "Weight vector", beta);
    check_finite(function, "Intercept", alpha);
    check_finite(function, "Matrix of independent variables", x);
  }
  // Compute log probability.
  T_partials_return logp(0.0);
  if (include_summand<propto>::value)
    logp += NEG_LOG_SQRT_TWO_PI * N;
  if (include_summand<propto, T_scale>::value) {
    if (is_vector<T_scale>::value) {
#ifdef STAN_OPENCL
      VectorXd log_sigma_partial_sum(wgs);
      copy(log_sigma_partial_sum, log_sigma_sum_cl);
      logp -= sum(log_sigma_partial_sum);
#else
      logp -= sum(log(sigma_val_vec));
#endif
    }
    else
      logp -= N * log(as_scalar(sigma_val));
  }
  if (include_summand<propto, T_y, T_x, T_alpha, T_beta, T_scale>::value)
    logp -= 0.5 * y_minus_mu_over_sigma_squared_sum;
  return ops_partials.build(logp);
}

template <typename T_y, typename T_x, typename T_alpha, typename T_beta,
          typename T_scale>
inline typename return_type<T_y, T_x, T_alpha, T_beta, T_scale>::type
normal_id_glm_lpdf(const T_y &y, const T_x &x, const T_alpha &alpha,
                   const T_beta &beta, const T_scale &sigma) {
  return normal_id_glm_lpdf<false>(y, x, alpha, beta, sigma);
}
}  // namespace math
}  // namespace stan
#endif
