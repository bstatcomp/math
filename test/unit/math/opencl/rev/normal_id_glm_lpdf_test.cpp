#ifdef STAN_OPENCL
#include <stan/math/opencl/rev/opencl.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>

using Eigen::Array;
using Eigen::Dynamic;
using Eigen::Matrix;
using stan::math::matrix_cl;
using stan::math::var;
using stan::test::expect_near_rel;

TEST(ProbDistributionsNormalIdGLM, error_checking) {
  int N = 3;
  int M = 2;

  Matrix<double, Dynamic, 1> y(N, 1);
  y << 51, 32, 12;
  Matrix<double, Dynamic, 1> y_size(N + 1, 1);
  y_size << 51, 32, 12, 34;
  Matrix<double, Dynamic, 1> y_value(N, 1);
  y_value << 51, 32, INFINITY;
  Matrix<double, Dynamic, Dynamic> x(N, M);
  x << -12, 46, -42, 24, 25, 27;
  Matrix<double, Dynamic, Dynamic> x_size1(N - 1, M);
  x_size1 << -12, 46, -42, 24;
  Matrix<double, Dynamic, Dynamic> x_size2(N, M - 1);
  x_size2 << -12, 46, -42;
  Matrix<double, Dynamic, Dynamic> x_value(N, M);
  x_value << -12, 46, -42, 24, 25, -INFINITY;
  Matrix<double, Dynamic, 1> beta(M, 1);
  beta << 0.3, 2;
  Matrix<double, Dynamic, 1> beta_size(M + 1, 1);
  beta_size << 0.3, 2, 0.4;
  Matrix<double, Dynamic, 1> beta_value(M, 1);
  beta_value << 0.3, INFINITY;
  Matrix<double, Dynamic, 1> alpha(N, 1);
  alpha << 0.3, -0.8, 1.8;
  Matrix<double, Dynamic, 1> alpha_size(N - 1, 1);
  alpha_size << 0.3, -0.8;
  Matrix<double, Dynamic, 1> alpha_value(N, 1);
  alpha_value << 0.3, -0.8, NAN;
  Matrix<double, Dynamic, 1> sigma(N, 1);
  sigma << 5, 2, 3.4;
  Matrix<double, Dynamic, 1> sigma_size(N + 1, 1);
  sigma_size << 5, 2, 3.4, 6;
  Matrix<double, Dynamic, 1> sigma_value(N, 1);
  sigma_value << 5, 2, NAN;

  matrix_cl<double> x_cl(x);
  matrix_cl<double> x_size1_cl(x_size1);
  matrix_cl<double> x_size2_cl(x_size2);
  matrix_cl<double> x_value_cl(x_value);
  matrix_cl<double> y_cl(y);
  matrix_cl<double> y_size_cl(y_size);
  matrix_cl<double> y_value_cl(y_value);
  matrix_cl<double> beta_cl(beta);
  matrix_cl<double> beta_size_cl(beta_size);
  matrix_cl<double> beta_value_cl(beta_value);
  matrix_cl<double> alpha_cl(alpha);
  matrix_cl<double> alpha_size_cl(alpha_size);
  matrix_cl<double> alpha_value_cl(alpha_value);
  matrix_cl<double> sigma_cl(sigma);
  matrix_cl<double> sigma_size_cl(sigma_size);
  matrix_cl<double> sigma_value_cl(sigma_value);

  EXPECT_NO_THROW(
      stan::math::normal_id_glm_lpdf(y_cl, x_cl, alpha_cl, beta_cl, sigma_cl));

  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_size_cl, x_cl, alpha_cl,
                                              beta_cl, sigma_cl),
               std::invalid_argument);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_size1_cl, alpha_cl,
                                              beta_cl, sigma_cl),
               std::invalid_argument);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_size2_cl, alpha_cl,
                                              beta_cl, sigma_cl),
               std::invalid_argument);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_cl, alpha_size_cl,
                                              beta_cl, sigma_cl),
               std::invalid_argument);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_cl, alpha_cl,
                                              beta_size_cl, sigma_cl),
               std::invalid_argument);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_cl, alpha_cl, beta_cl,
                                              sigma_size_cl),
               std::invalid_argument);

  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_value_cl, x_cl, alpha_cl,
                                              beta_cl, sigma_cl),
               std::domain_error);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_value_cl, alpha_cl,
                                              beta_cl, sigma_cl),
               std::domain_error);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_cl, alpha_value_cl,
                                              beta_cl, sigma_cl),
               std::domain_error);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_cl, alpha_cl,
                                              beta_value_cl, sigma_cl),
               std::domain_error);
  EXPECT_THROW(stan::math::normal_id_glm_lpdf(y_cl, x_cl, alpha_cl, beta_cl,
                                              sigma_value_cl),
               std::domain_error);
}

auto normal_id_glm_lpdf_functor
    = [](const auto& y, const auto& x, const auto& alpha, const auto& beta,
         const auto& sigma) {
        return stan::math::normal_id_glm_lpdf(y, x, alpha, beta, sigma);
      };
auto normal_id_glm_lpdf_functor_propto
    = [](const auto& y, const auto& x, const auto& alpha, const auto& beta,
         const auto& sigma) {
        return stan::math::normal_id_glm_lpdf<true>(y, x, alpha, beta, sigma);
      };

TEST(ProbDistributionsNormalIdGLM, opencl_matches_cpu_small_simple) {
  int N = 3;
  int M = 2;

  Matrix<double, Dynamic, 1> y(N, 1);
  y << 51, 32, 12;
  Matrix<double, Dynamic, Dynamic> x(N, M);
  x << -12, 46, -42, 24, 25, 27;
  Matrix<double, Dynamic, 1> beta(M, 1);
  beta << 0.3, 2;
  double alpha = 0.3;
  double sigma = 11;

  stan::math::test::compare_cpu_opencl_prim_rev(normal_id_glm_lpdf_functor, y,
                                                x, alpha, beta, sigma);
  stan::math::test::compare_cpu_opencl_prim_rev(
      normal_id_glm_lpdf_functor_propto, y, x, alpha, beta, sigma);
}

TEST(ProbDistributionsNormalIdGLM, opencl_broadcast_y) {
  int N = 3;
  int M = 2;

  double y = 13;
  Matrix<double, Dynamic, 1> y_vec
      = Matrix<double, Dynamic, 1>::Constant(N, 1, y);
  Matrix<double, Dynamic, Dynamic> x(N, M);
  x << -12, 46, -42, 24, 25, 27;
  Matrix<double, Dynamic, 1> beta(M, 1);
  beta << 0.3, 2;
  double alpha = 0.3;
  double sigma = 11;

  matrix_cl<double> x_cl(x);
  matrix_cl<double> y_vec_cl(y_vec);
  matrix_cl<double> beta_cl(beta);

  expect_near_rel(
      "normal_id_glm_lpdf (OpenCL)",
      stan::math::normal_id_glm_lpdf(y, x_cl, alpha, beta_cl, sigma),
      stan::math::normal_id_glm_lpdf(y_vec_cl, x_cl, alpha, beta_cl, sigma));
  expect_near_rel(
      "normal_id_glm_lpdf (OpenCL)",
      stan::math::normal_id_glm_lpdf<true>(y, x_cl, alpha, beta_cl, sigma),
      stan::math::normal_id_glm_lpdf<true>(y_vec_cl, x_cl, alpha, beta_cl,
                                           sigma));

  var y_var1 = y;
  Matrix<var, Dynamic, 1> y_var2 = y_vec;
  Matrix<var, Dynamic, Dynamic> x_var1 = x;
  Matrix<var, Dynamic, Dynamic> x_var2 = x;
  Matrix<var, Dynamic, 1> beta_var1 = beta;
  Matrix<var, Dynamic, 1> beta_var2 = beta;
  auto x_var1_cl = to_matrix_cl(x_var1);
  auto x_var2_cl = to_matrix_cl(x_var2);
  auto y_var2_cl = to_matrix_cl(y_var2);
  auto beta_var1_cl = to_matrix_cl(beta_var1);
  auto beta_var2_cl = to_matrix_cl(beta_var2);
  var alpha_var1 = alpha;
  var alpha_var2 = alpha;
  var sigma_var1 = sigma;
  var sigma_var2 = sigma;

  var res1 = stan::math::normal_id_glm_lpdf(y_var1, x_var1_cl, alpha_var1,
                                            beta_var1_cl, sigma_var1);
  var res2 = stan::math::normal_id_glm_lpdf(y_var2_cl, x_var2_cl, alpha_var2,
                                            beta_var2_cl, sigma_var2);

  (res1 + res2).grad();

  expect_near_rel("normal_id_glm_lpdf (OpenCL)", res1.val(), res2.val());

  expect_near_rel("normal_id_glm_lpdf (OpenCL)", x_var1.adj().eval(),
                  x_var2.adj().eval());
  expect_near_rel("normal_id_glm_lpdf (OpenCL)", alpha_var1.adj(),
                  alpha_var2.adj());
  expect_near_rel("normal_id_glm_lpdf (OpenCL)", sigma_var1.adj(),
                  sigma_var2.adj());
  expect_near_rel("normal_id_glm_lpdf (OpenCL)", beta_var1.adj().eval(),
                  beta_var2.adj().eval());
}

TEST(ProbDistributionsNormalIdGLM, opencl_matches_cpu_zero_instances) {
  int N = 0;
  int M = 2;

  Matrix<double, Dynamic, 1> y(N, 1);
  Matrix<double, Dynamic, Dynamic> x(N, M);
  Matrix<double, Dynamic, 1> beta(M, 1);
  beta << 0.3, 2;
  double alpha = 0.3;
  double sigma = 11;

  stan::math::test::compare_cpu_opencl_prim_rev(normal_id_glm_lpdf_functor, y,
                                                x, alpha, beta, sigma);
  stan::math::test::compare_cpu_opencl_prim_rev(
      normal_id_glm_lpdf_functor_propto, y, x, alpha, beta, sigma);
}

TEST(ProbDistributionsNormalIdGLM, opencl_matches_cpu_zero_attributes) {
  int N = 3;
  int M = 0;

  Matrix<double, Dynamic, 1> y(N, 1);
  y << 51, 32, 12;
  Matrix<double, Dynamic, Dynamic> x(N, M);
  Matrix<double, Dynamic, 1> beta(M, 1);
  double alpha = 0.3;
  double sigma = 11;

  stan::math::test::compare_cpu_opencl_prim_rev(normal_id_glm_lpdf_functor, y,
                                                x, alpha, beta, sigma);
  stan::math::test::compare_cpu_opencl_prim_rev(
      normal_id_glm_lpdf_functor_propto, y, x, alpha, beta, sigma);
}

TEST(ProbDistributionsNormalIdGLM,
     opencl_matches_cpu_small_vector_alpha_sigma) {
  int N = 3;
  int M = 2;

  Matrix<double, Dynamic, 1> y(N, 1);
  y << 51, 32, 12;
  Matrix<double, Dynamic, Dynamic> x(N, M);
  x << -12, 46, -42, 24, 25, 27;
  Matrix<double, Dynamic, 1> beta(M, 1);
  beta << 0.3, 2;
  Matrix<double, Dynamic, 1> alpha(N, 1);
  alpha << 0.3, -0.8, 1.8;
  Matrix<double, Dynamic, 1> sigma(N, 1);
  sigma << 5, 2, 3.4;

  stan::math::test::compare_cpu_opencl_prim_rev(normal_id_glm_lpdf_functor, y,
                                                x, alpha, beta, sigma);
  stan::math::test::compare_cpu_opencl_prim_rev(
      normal_id_glm_lpdf_functor_propto, y, x, alpha, beta, sigma);
}

TEST(ProbDistributionsNormalIdGLM, opencl_matches_cpu_big) {
  int N = 153;
  int M = 71;

  Matrix<double, Dynamic, 1> y = Matrix<double, Dynamic, 1>::Random(N, 1);
  Matrix<double, Dynamic, Dynamic> x
      = Matrix<double, Dynamic, Dynamic>::Random(N, M);
  Matrix<double, Dynamic, 1> beta = Matrix<double, Dynamic, 1>::Random(M, 1);
  Matrix<double, Dynamic, 1> alpha = Matrix<double, Dynamic, 1>::Random(N, 1);
  Matrix<double, Dynamic, 1> sigma
      = Array<double, Dynamic, 1>::Random(N, 1) + 1.1;

  stan::math::test::compare_cpu_opencl_prim_rev(normal_id_glm_lpdf_functor, y,
                                                x, alpha, beta, sigma);
  stan::math::test::compare_cpu_opencl_prim_rev(
      normal_id_glm_lpdf_functor_propto, y, x, alpha, beta, sigma);
}

#endif
