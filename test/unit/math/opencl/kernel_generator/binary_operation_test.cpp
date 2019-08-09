#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/binary_operation.hpp>
#include <stan/math/opencl/kernel_generator/load.hpp>
//#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using stan::math::matrix_cl;

TEST(MathMatrixCL, addition_test){
  using stan::math::load;
  using stan::math::addition;
  MatrixXd m1(3, 3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  MatrixXi m2(3, 3);
  m2 << 10, 100, 1000, 0, -10, -12, 2, 4, 8;
  
  matrix_cl<double> m1_cl(m1);
  matrix_cl<int> m2_cl(m2);
  matrix_cl<double> res_cl = addition<double, int>(load<double>(m1_cl), load<int>(m2_cl));

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

TEST(MathMatrixCL, tmp){

}

#endif