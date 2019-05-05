#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>
#ifdef STAN_OPENCL
#include <stan/math/opencl/opencl_context.hpp>
#include <boost/random/mersenne_twister.hpp>
#endif
TEST(MathMatrix, mdivide_left_tri_val) {
  using stan::math::mdivide_left_tri;
  stan::math::matrix_d Ad(2, 2);
  stan::math::matrix_d Ad_inv(2, 2);
  stan::math::matrix_d I;

  Ad << 2.0, 0.0, 5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Ad, Ad);
  EXPECT_NEAR(1.0, I(0, 0), 1.0E-12);
  EXPECT_NEAR(0.0, I(0, 1), 1.0E-12);
  EXPECT_NEAR(0.0, I(1, 0), 1.0E-12);
  EXPECT_NEAR(1.0, I(1, 1), 1.0e-12);

  Ad_inv = mdivide_left_tri<Eigen::Lower>(Ad);
  I = Ad * Ad_inv;
  EXPECT_NEAR(1.0, I(0, 0), 1.0E-12);
  EXPECT_NEAR(0.0, I(0, 1), 1.0E-12);
  EXPECT_NEAR(0.0, I(1, 0), 1.0E-12);
  EXPECT_NEAR(1.0, I(1, 1), 1.0e-12);

  Ad << 2.0, 3.0, 0.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Ad, Ad);
  EXPECT_NEAR(1.0, I(0, 0), 1.0E-12);
  EXPECT_NEAR(0.0, I(0, 1), 1.0E-12);
  EXPECT_NEAR(0.0, I(1, 0), 1.0E-12);
  EXPECT_NEAR(1.0, I(1, 1), 1.0e-12);
}

#ifdef STAN_OPENCL
#define EXPECT_MATRIX_NEAR(A, B, DELTA) \
  for (int i = 0; i < A.size(); i++)    \
    EXPECT_NEAR(A(i), B(i), DELTA);
TEST(MathMatrix, mdivide_left_tri_val_lower_opencl) {
  using stan::math::mdivide_left_tri;
  boost::random::mt19937 rng;
  
  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1;
  
  int size = 512;
  
  auto m1 = stan::math::matrix_d::Zero(size, size).eval();
  for(int i= 0;i<size;i++){
    for(int j= 0;j<i;j++){
      m1(i, j) = stan::math::normal_rng(-2, 2, rng);
    }
    m1(i,i) = 100;
  }
  
  auto m1_inv_cl = stan::math::mdivide_left_tri<Eigen::Lower>(m1);
  // to make sure we dont use OpenCL
  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1024;

  auto m1_inv = stan::math::mdivide_left_tri<Eigen::Lower>(m1);
  EXPECT_MATRIX_NEAR(m1_inv, m1_inv_cl, 1e-10);
}

TEST(MathMatrix, mdivide_left_tri_val_opencl) {
  using stan::math::mdivide_left_tri;
  boost::random::mt19937 rng;
  
  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1;
  
  int size = 512;
  
  auto m1 = stan::math::matrix_d::Zero(size, size).eval();
  for(int i= 0;i<size;i++){
    for(int j= 0;j<i;j++){
      m1(i, j) = stan::math::normal_rng(-2, 2, rng);
    }
    m1(i,i) = 100;
  }
  
  auto m1_inv_cl = stan::math::mdivide_left_tri<Eigen::Upper>(m1);
  // to make sure we dont use OpenCL
  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1024;

  auto m1_inv = stan::math::mdivide_left_tri<Eigen::Upper>(m1);
  EXPECT_MATRIX_NEAR(m1_inv, m1_inv_cl, 1e-10);
}

TEST(MathMatrix, mdivide_left_tri_right_side_lower_opencl) {
  using stan::math::mdivide_left_tri;
  boost::random::mt19937 rng;  
  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1;
  
  int size = 512;
  
  auto m1 = stan::math::matrix_d::Zero(size, size).eval();
  auto m2 = stan::math::matrix_d::Identity(size, size).eval();
  for(int i= 0;i<size;i++){
    for(int j= 0;j<i;j++){
      m1(i, j) = stan::math::normal_rng(-2, 2, rng);
    }
    m1(i,i) = 100;
  }
  
  auto m1_inv_cl = stan::math::mdivide_left_tri<Eigen::Lower>(m1, m1);

  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1000;

  auto m1_inv = stan::math::mdivide_left_tri<Eigen::Lower>(m1, m1);
    
  EXPECT_MATRIX_NEAR(m1_inv, m1_inv_cl, 1e-10);
}

TEST(MathMatrix, mdivide_left_tri_right_side_upper_opencl) {
  using stan::math::mdivide_left_tri;
  boost::random::mt19937 rng;  
  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1;
  
  int size = 512;
  
  auto m1 = stan::math::matrix_d::Zero(size, size).eval();
  auto m2 = stan::math::matrix_d::Identity(size, size).eval();
  for(int i= 0;i<size;i++){
    for(int j= 0;j<i;j++){
      m1(i, j) = stan::math::normal_rng(-2, 2, rng);
    }
    m1(i,i) = 100;
  }
  
  auto m1_inv_cl = stan::math::mdivide_left_tri<Eigen::Upper>(m1, m1);

  stan::math::opencl_context.tuning_opts().lower_tri_inverse_size_worth_transfer = 1000;

  auto m1_inv = stan::math::mdivide_left_tri<Eigen::Upper>(m1, m1);
    
  EXPECT_MATRIX_NEAR(m1_inv, m1_inv_cl, 1e-10);
}
#endif
