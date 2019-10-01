#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/binary_operation.hpp>
#include <stan/math/opencl/kernel_generator/multi_result_kernel.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using stan::math::matrix_cl;

#define EXPECT_MATRIX_NEAR(A, B, DELTA) \
  for (int i = 0; i < A.size(); i++)    \
    EXPECT_NEAR(A(i), B(i), DELTA);

TEST(MathMatrixCL, multi_result_kernel){
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXd m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<double> m2_cl(m2);

  matrix_cl<double> sum_cl(3, 3);
  matrix_cl<double> diff_cl(3, 3);

  auto sum = m1_cl + m2_cl;
  stan::math::results(sum_cl,diff_cl) = stan::math::expressions(sum, m1_cl - m2_cl);

  MatrixXd res_sum = stan::math::from_matrix_cl(sum_cl);
  MatrixXd res_diff = stan::math::from_matrix_cl(diff_cl);

  MatrixXd correct_sum = m1 + m2;
  MatrixXd correct_diff = m1 - m2;

  EXPECT_MATRIX_NEAR(res_sum,correct_sum,1e-9);
  EXPECT_MATRIX_NEAR(res_diff,correct_diff,1e-9);
}

TEST(MathMatrixCL, multi_result_kernel_reuse_kernel){
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXd m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<double> m2_cl(m2);

  matrix_cl<double> sum1_cl(3, 3);
  matrix_cl<double> diff1_cl(3, 3);
  matrix_cl<double> sum2_cl(3, 3);
  matrix_cl<double> diff2_cl(3, 3);

  auto sum = m1_cl + m2_cl;
  stan::math::results(sum1_cl,diff1_cl) = stan::math::expressions(sum, m1_cl - m2_cl);
  stan::math::results(sum2_cl,diff2_cl) = stan::math::expressions(sum, m1_cl - m2_cl);

  MatrixXd res_sum = stan::math::from_matrix_cl(sum2_cl);
  MatrixXd res_diff = stan::math::from_matrix_cl(diff2_cl);

  MatrixXd correct_sum = m1 + m2;
  MatrixXd correct_diff = m1 - m2;

  EXPECT_MATRIX_NEAR(res_sum,correct_sum,1e-9);
  EXPECT_MATRIX_NEAR(res_diff,correct_diff,1e-9);
}

#endif
