#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <test/unit/math/opencl/kernel_generator/reference_kernel.hpp>
#include <test/unit/util.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <string>

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using stan::math::matrix_cl;

TEST(KernelGenerator, reshape_errors) {
  using stan::math::reshape;

  matrix_cl<double> m(7, 9);

  EXPECT_NO_THROW(reshape(m, 9, 7));
  EXPECT_NO_THROW(reshape(m, 3, 21));
  EXPECT_THROW(reshape(m, 0,0), std::invalid_argument);
  EXPECT_THROW(reshape(m, 0, 5), std::invalid_argument);
  EXPECT_THROW(reshape(m, 7, 7), std::invalid_argument);
  EXPECT_THROW(reshape(m, 9, 9), std::invalid_argument);
}

TEST(KernelGenerator, reshape_test) {
  using stan::math::reshape;
  MatrixXd m = MatrixXd::Random(7, 9);

  matrix_cl<double> m_cl(m);

  matrix_cl<double> res_cl = reshape(m_cl, 21, 3);
  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  MatrixXd correct = Eigen::Map<Eigen::MatrixXd>(m.data(), 21, 3);
  EXPECT_MATRIX_NEAR(res, correct, 1e-9);
}

TEST(KernelGenerator, reshape_multiple_operations_test) {
  using stan::math::reshape;
  MatrixXd m = MatrixXd::Random(7, 9);

  matrix_cl<double> m_cl(m);

  auto tmp = reshape(reshape(m_cl, 9, 7), 21, 3);
  matrix_cl<double> res_cl = tmp;
  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  MatrixXd correct = Eigen::Map<Eigen::MatrixXd>(m.data(), 21, 3);
  EXPECT_MATRIX_NEAR(res, correct, 1e-9);
}

TEST(KernelGenerator, reshape_multiple_operations_lvalue_test) {
  using stan::math::reshape;
  MatrixXd m = MatrixXd::Random(7, 9);

  matrix_cl<double> m_cl(m);

  auto tmp2 = reshape(m_cl, 9, 7);
  auto tmp = reshape(tmp2, 21, 3);
  matrix_cl<double> res_cl = tmp;
  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  MatrixXd correct = Eigen::Map<Eigen::MatrixXd>(m.data(), 21, 3);
  EXPECT_MATRIX_NEAR(res, correct, 1e-9);
}

TEST(KernelGenerator, colwise_sum_reshape_test) {
  using stan::math::reshape;
  int N = 4096;
  int M = 4096;
  MatrixXd m = MatrixXd::Random(N,M);

  matrix_cl<double> m_cl(m);

  matrix_cl<double> res_cl = colwise_sum(reshape(m_cl, N*M, 1));
  MatrixXd raw_res = stan::math::from_matrix_cl(res_cl);
  MatrixXd res = raw_res.colwise().sum();

  MatrixXd correct = Eigen::Map<Eigen::MatrixXd>(m.data(), N*M, 1).colwise().sum();
  EXPECT_MATRIX_NEAR(res, correct, 1e-9);
}

#endif
