#ifdef STAN_OPENCL

//#include <stan/math/opencl/add.hpp>
#include <stan/math/opencl/kernel_generator/binary_operation.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using stan::math::matrix_cl;

#define EXPECT_MATRIX_NEAR(A, B, DELTA) \
  for (int i = 0; i < A.size(); i++)    \
    EXPECT_NEAR(A(i), B(i), DELTA);

TEST(MathMatrixCL, addition_test){
  MatrixXd m1(3, 3);
  m1 << 1, 2.5, 3, 4, 5, 6.3, 7, -8, -9.5;
  MatrixXd m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;
  
  matrix_cl<double> m1_cl(m1);
  matrix_cl<double> m2_cl(m2);
  auto tmp = m1_cl + m2_cl;
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_MATRIX_NEAR(res, (m1+m2.cast<double>()).eval(),1e-9);
}

TEST(MathMatrixCL, subtraction_test){
  MatrixXd m1(3, 3);
  m1 << 1, 2.5, 3, 4, 5, 6.3, 7, -8, -9.5;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  auto tmp = m1_cl - m2_cl;
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_MATRIX_NEAR(res, (m1-m2.cast<double>()).eval(),1e-9);
}

TEST(MathMatrixCL, elewise_multiplication_test){
  MatrixXd m1(3, 3);
  m1 << 1, 2.5, 3, 4, 5, 6.3, 7, -8, -9.5;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  auto tmp = elewise_multiplication(m1_cl, m2_cl);
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_MATRIX_NEAR(res, (m1.array()*m2.cast<double>().array()).matrix().eval(),1e-9);
}


TEST(MathMatrixCL, elewise_division_test){
  MatrixXd m1(3, 3);
  m1 << 1, 2.5, 3, 4, 5, 6.3, 7, -8, -9.5;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 1, -10, -12, 2, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  auto tmp = elewise_division(m1_cl, m2_cl);
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);

  EXPECT_MATRIX_NEAR(res, (m1.array()/m2.cast<double>().array()).matrix().eval(),1e-9);
}

TEST(MathMatrixCL, multiple_operations){
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;
  MatrixXi m3(3, 3);
  m3 << 1, 10, 1100, -40, -14, -1, 2.5, 4, 8;

  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  matrix_cl<int> m3_cl(m3);
  auto tmp = m1_cl - (m2_cl + m3_cl);
  matrix_cl<double> res_cl = tmp;

  MatrixXd res = stan::math::from_matrix_cl(res_cl);


  EXPECT_MATRIX_NEAR(res, (m1-(m2+m3).cast<double>()).eval(),1e-9);
}

#endif