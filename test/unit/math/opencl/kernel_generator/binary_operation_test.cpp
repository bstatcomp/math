#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/binary_operation.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using stan::math::matrix_cl;

TEST(MathMatrixCL, addition_test){
  using stan::math::addition;
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;
  
  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  auto tmp = addition(m1_cl, m2_cl);
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_EQ(11, res(0, 0));
  EXPECT_EQ(102, res(0, 1));
  EXPECT_EQ(1003, res(0, 2));
  EXPECT_EQ(4, res(1, 0));
  EXPECT_EQ(-5, res(1, 1));
  EXPECT_EQ(-6, res(1, 2));
  EXPECT_EQ(9, res(2, 0));
  EXPECT_EQ(12, res(2, 1));
  EXPECT_EQ(17, res(2, 2));
}

TEST(MathMatrixCL, subtraction_test){
  using stan::math::subtraction;
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  auto tmp = subtraction(m1_cl, m2_cl);
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_EQ(-9, res(0, 0));
  EXPECT_EQ(-98, res(0, 1));
  EXPECT_EQ(-997, res(0, 2));
  EXPECT_EQ(4, res(1, 0));
  EXPECT_EQ(15, res(1, 1));
  EXPECT_EQ(18, res(1, 2));
  EXPECT_EQ(5, res(2, 0));
  EXPECT_EQ(4, res(2, 1));
  EXPECT_EQ(1, res(2, 2));
}

TEST(MathMatrixCL, elewise_multiplication_test){
  using stan::math::elewise_multiplication;
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  auto tmp = elewise_multiplication(m1_cl, m2_cl);
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_EQ(10, res(0, 0));
  EXPECT_EQ(200, res(0, 1));
  EXPECT_EQ(3000, res(0, 2));
  EXPECT_EQ(0, res(1, 0));
  EXPECT_EQ(-50, res(1, 1));
  EXPECT_EQ(-72, res(1, 2));
  EXPECT_EQ(14, res(2, 0));
  EXPECT_EQ(32, res(2, 1));
  EXPECT_EQ(72, res(2, 2));
}


TEST(MathMatrixCL, elewise_division_test){
  using stan::math::elewise_division;
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  auto tmp = elewise_division(m1_cl, m2_cl);
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_EQ(0.1, res(0, 0));
  EXPECT_EQ(0.02, res(0, 1));
  EXPECT_EQ(0.003, res(0, 2));
  EXPECT_EQ(4./0, res(1, 0));
  EXPECT_EQ(-0.5, res(1, 1));
  EXPECT_EQ(-0.5, res(1, 2));
  EXPECT_EQ(3.5, res(2, 0));
  EXPECT_EQ(2, res(2, 1));
  EXPECT_EQ(9./8, res(2, 2));
}

#endif