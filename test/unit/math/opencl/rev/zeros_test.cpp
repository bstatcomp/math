#ifdef STAN_OPENCL
#include <stan/math/prim/mat.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/rev/zeros.hpp>
#include <gtest/gtest.h>
#include <algorithm>

TEST(MathMatrixCL, zero_m_exception_pass) {
  using stan::math::var;
  stan::math::matrix_cl<var> m(1, 1);
  m.zeros();
}
/*
TEST(MathMatrixCL, zero_m_value_check) {
  stan::math::matrix_d m0(2, 2);
  stan::math::matrix_d m0_dst(2, 2);
  m0 << 2, 2, 2, 2;
  stan::math::matrix_cl<double> m(m0);
  stan::math::matrix_cl<double> m_upper(m0);
  stan::math::matrix_cl<double> m_lower(m0);

  EXPECT_NO_THROW(m.zeros<stan::math::TriangularViewCL::Entire, stan::enable_if_var_or_vari<var>>());
  EXPECT_NO_THROW(m_lower.zeros<stan::math::TriangularViewCL::Lower, stan::enable_if_var_or_vari<var>>());
  EXPECT_NO_THROW(m_upper.zeros<stan::math::TriangularViewCL::Upper, stan::enable_if_var_or_vari<var>>());

  m0_dst = stan::math::from_matrix_cl(m);
  EXPECT_EQ(0, m0_dst(0, 0));
  EXPECT_EQ(0, m0_dst(0, 1));
  EXPECT_EQ(0, m0_dst(1, 0));
  EXPECT_EQ(0, m0_dst(1, 1));

  m0_dst = stan::math::from_matrix_cl(m_lower);
  EXPECT_EQ(2, m0_dst(0, 0));
  EXPECT_EQ(2, m0_dst(0, 1));
  EXPECT_EQ(0, m0_dst(1, 0));
  EXPECT_EQ(2, m0_dst(1, 1));

  m0_dst = stan::math::from_matrix_cl(m_upper);
  EXPECT_EQ(2, m0_dst(0, 0));
  EXPECT_EQ(0, m0_dst(0, 1));
  EXPECT_EQ(2, m0_dst(1, 0));
  EXPECT_EQ(2, m0_dst(1, 1));
}
*/
#endif
