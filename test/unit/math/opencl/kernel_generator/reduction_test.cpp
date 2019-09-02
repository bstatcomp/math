#ifdef STAN_OPENCL

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/opencl/kernel_generator/reduction.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>

using Eigen::Dynamic;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using stan::math::matrix_cl;

#define EXPECT_MATRIX_NEAR(A, B, DELTA) \
  for (int i = 0; i < A.size(); i++)    \
    EXPECT_NEAR(A(i), B(i), DELTA);

TEST(MathMatrixCL, sum_test){
  MatrixXd m(3, 2);
  m << 1.1, 1.2,
      1.3, 1.4,
      1.5, 1.6;

  matrix_cl<double> m_cl(m);

  matrix_cl<double> res1_cl = stan::math::sum<true,false>(m_cl);
  MatrixXd res1 = stan::math::from_matrix_cl(res1_cl);
  MatrixXd correct1 = m.rowwise().sum();
  EXPECT_EQ(correct1.rows(), res1.rows());
  EXPECT_EQ(correct1.cols(), res1.cols());
  EXPECT_MATRIX_NEAR(correct1,res1,1e-9);

  matrix_cl<double> res2_cl = stan::math::sum<false,true>(m_cl);
  MatrixXd res2 = stan::math::from_matrix_cl(res2_cl);
  MatrixXd correct2 = m.colwise().sum();
  EXPECT_EQ(correct2.rows(), res2.rows());
  EXPECT_EQ(correct2.cols(), res2.cols());
  EXPECT_MATRIX_NEAR(correct2,res2,1e-9);
}

TEST(MathMatrixCL, min_test){
  MatrixXd m(3, 2);
  m << 1.1, 1.2,
          1.3, 1.4,
          1.5, 1.6;

  matrix_cl<double> m_cl(m);

  matrix_cl<double> res1_cl = stan::math::min<true,false>(m_cl);
  MatrixXd res1 = stan::math::from_matrix_cl(res1_cl);
  MatrixXd correct1 = m.rowwise().minCoeff();
  EXPECT_EQ(correct1.rows(), res1.rows());
  EXPECT_EQ(correct1.cols(), res1.cols());
  EXPECT_MATRIX_NEAR(correct1,res1,1e-9);

  matrix_cl<double> res2_cl = stan::math::min<false,true>(m_cl);
  MatrixXd res2 = stan::math::from_matrix_cl(res2_cl);
  MatrixXd correct2 = m.colwise().minCoeff();
  EXPECT_EQ(correct2.rows(), res2.rows());
  EXPECT_EQ(correct2.cols(), res2.cols());
  EXPECT_MATRIX_NEAR(correct2,res2,1e-9);
}

TEST(MathMatrixCL, max_test){
  MatrixXd m(3, 2);
  m << 1.1, 1.2,
          1.3, 1.4,
          1.5, 1.6;

  matrix_cl<double> m_cl(m);

  matrix_cl<double> res1_cl = stan::math::max<true,false>(m_cl);
  MatrixXd res1 = stan::math::from_matrix_cl(res1_cl);
  MatrixXd correct1 = m.rowwise().maxCoeff();
  EXPECT_EQ(correct1.rows(), res1.rows());
  EXPECT_EQ(correct1.cols(), res1.cols());
  EXPECT_MATRIX_NEAR(correct1,res1,1e-9);

  matrix_cl<double> res2_cl = stan::math::max<false,true>(m_cl);
  MatrixXd res2 = stan::math::from_matrix_cl(res2_cl);
  MatrixXd correct2 = m.colwise().maxCoeff();
  EXPECT_EQ(correct2.rows(), res2.rows());
  EXPECT_EQ(correct2.cols(), res2.cols());
  EXPECT_MATRIX_NEAR(correct2,res2,1e-9);
}

TEST(MathMatrixCL, sum_triangular_test){
  MatrixXd m(3, 2);
  m << 1.1, 1.2,
          1.3, 1.4,
          1.5, 1.6;

  matrix_cl<double> m_cl(m, stan::math::matrix_cl_view::Lower);

  MatrixXd m12  = m;
  m12.triangularView<Eigen::StrictlyUpper>() = MatrixXd::Constant(3,2,0);

  matrix_cl<double> res1_cl = stan::math::sum<true,false>(m_cl);
  MatrixXd res1 = stan::math::from_matrix_cl(res1_cl);
  MatrixXd correct1 = m12.rowwise().sum();
  EXPECT_EQ(correct1.rows(), res1.rows());
  EXPECT_EQ(correct1.cols(), res1.cols());
  EXPECT_MATRIX_NEAR(correct1,res1,1e-9);

  matrix_cl<double> res2_cl = stan::math::sum<false,true>(m_cl);
  MatrixXd res2 = stan::math::from_matrix_cl(res2_cl);
  MatrixXd correct2 = m12.colwise().sum();
  EXPECT_EQ(correct2.rows(), res2.rows());
  EXPECT_EQ(correct2.cols(), res2.cols());
  EXPECT_MATRIX_NEAR(correct2,res2,1e-9);

  m_cl.view(stan::math::matrix_cl_view::Upper);

  MatrixXd m34  = m;
  m34.triangularView<Eigen::StrictlyLower>() = MatrixXd::Constant(3,2,0);

  matrix_cl<double> res3_cl = stan::math::sum<true,false>(m_cl);
  MatrixXd res3 = stan::math::from_matrix_cl(res3_cl);
  MatrixXd correct3 = m34.rowwise().sum();
  EXPECT_EQ(correct3.rows(), res3.rows());
  EXPECT_EQ(correct3.cols(), res3.cols());
  EXPECT_MATRIX_NEAR(correct3,res3,1e-9);

  matrix_cl<double> res4_cl = stan::math::sum<false,true>(m_cl);
  MatrixXd res4 = stan::math::from_matrix_cl(res4_cl);
  MatrixXd correct4 = m34.colwise().sum();
  EXPECT_EQ(correct4.rows(), res4.rows());
  EXPECT_EQ(correct4.cols(), res4.cols());
  EXPECT_MATRIX_NEAR(correct4,res4,1e-9);
}

#endif