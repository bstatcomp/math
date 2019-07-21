#ifdef STAN_OPENCL
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/add.hpp>
#include <stan/math/opencl/rev/sub_block.hpp>
#include <gtest/gtest.h>
#include <algorithm>

TEST(MathMatrixVarCL, sub_block_pass_vari) {
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::var;
  using stan::math::matrix_cl;
  using stan::math::vari;
  using stan::math::matrix_vi;
  matrix_v d1(3, 3);
  d1 << 9,8,7,6,5,4,3,2,1;
  matrix_v d2(4, 4);
  d2 << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;

  stan::math::matrix_d m1(4, 4);
  m1 << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16;
  matrix_cl<var> d11(d1);
  matrix_cl<var> d22(d2);
  matrix_cl<double> d33(m1);
  d22.sub_block(d11, 0, 0, 0, 0, 2, 2);
  matrix_d d3 = stan::math::from_matrix_cl(d22.val());

  EXPECT_EQ(9, d3(0, 0));
  EXPECT_EQ(8, d3(0, 1));
  EXPECT_EQ(3, d3(0, 2));

  EXPECT_EQ(6, d3(1, 0));
  EXPECT_EQ(5, d3(1, 1));
  EXPECT_EQ(7, d3(1, 2));
}


TEST(MathMatrixVarCL, sub_block_exception) {
  using stan::math::var;
  stan::math::matrix_v d1;
  stan::math::matrix_v d2;

  d1.resize(3, 3);
  d2.resize(4, 4);
  stan::math::matrix_cl<var> d11(d1);
  stan::math::matrix_cl<var> d22(d2);
  EXPECT_THROW(d22.sub_block(d11, 1, 1, 0, 0, 4, 4), std::domain_error);
  EXPECT_THROW(d22.sub_block(d11, 4, 4, 0, 0, 2, 2), std::domain_error);
}

TEST(MathMatrixVarCL, sub_block_lower_pass) {
  using stan::math::var;
  using stan::math::matrix_cl;
  using stan::math::matrix_v;
  using stan::math::matrix_d;
  matrix_v d1;
  matrix_v d2;

  d1.resize(3, 3);
  d2.resize(4, 4);

  d1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  d2 << 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1;

  matrix_cl<var> d11(d1);
  matrix_cl<var> d22(d2);
  d22.sub_block<stan::math::TriangularViewCL::Lower>(d11, 0, 0, 0, 0, 3, 3);
  matrix_d d3 = stan::math::from_matrix_cl(d22.val());
  EXPECT_EQ(1, d3(0, 0));
  EXPECT_EQ(15, d3(0, 1));
  EXPECT_EQ(14, d3(0, 2));
  EXPECT_EQ(10, d3(1, 2));
  EXPECT_EQ(7, d3(2, 0));
}

#endif
