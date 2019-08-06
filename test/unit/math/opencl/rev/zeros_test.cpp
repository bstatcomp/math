#ifdef STAN_OPENCL
#include <stan/math/prim/mat.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/rev/zeros.hpp>
#include <gtest/gtest.h>
#include <algorithm>

TEST(MathMatrixCL, zero_m_exception_pass) {
  using stan::math::matrix_cl;
  using stan::math::matrix_cl_view;
  using stan::math::var;
  matrix_cl<var> m(1, 1);

  EXPECT_NO_THROW(m.zeros<matrix_cl_view::Entire>());
  EXPECT_NO_THROW(m.zeros<matrix_cl_view::Lower>());
  EXPECT_NO_THROW(m.zeros<matrix_cl_view::Upper>());

  matrix_cl<double> m0;
  EXPECT_NO_THROW(m0.zeros<matrix_cl_view::Entire>());
  EXPECT_NO_THROW(m0.zeros<matrix_cl_view::Lower>());
  EXPECT_NO_THROW(m0.zeros<matrix_cl_view::Upper>());
}

TEST(MathMatrixCL, zero_m_value_check) {
  using stan::math::matrix_cl;
  using stan::math::matrix_cl_view;
  using stan::math::matrix_d;
  using stan::math::matrix_v;
  using stan::math::var;
  matrix_v m0(2, 2);
  m0 << 2, 2, 2, 2;
  matrix_cl<var> m(m0);
  matrix_cl<var> m_upper(m0);
  matrix_cl<var> m_lower(m0);

  EXPECT_NO_THROW(m.zeros<matrix_cl_view::Entire>());
  EXPECT_NO_THROW(m_lower.zeros<matrix_cl_view::Lower>());
  EXPECT_NO_THROW(m_upper.zeros<matrix_cl_view::Upper>());
  matrix_d m0_dst = from_matrix_cl(m.val());
  EXPECT_EQ(0, m0_dst(0, 0));
  EXPECT_EQ(0, m0_dst(0, 1));
  EXPECT_EQ(0, m0_dst(1, 0));
  EXPECT_EQ(0, m0_dst(1, 1));

  m0_dst = from_matrix_cl(m_lower.val());
  EXPECT_EQ(2, m0_dst(0, 0));
  EXPECT_EQ(2, m0_dst(0, 1));
  EXPECT_EQ(0, m0_dst(1, 0));
  EXPECT_EQ(2, m0_dst(1, 1));

  m0_dst = from_matrix_cl(m_upper.val());
  EXPECT_EQ(2, m0_dst(0, 0));
  EXPECT_EQ(0, m0_dst(0, 1));
  EXPECT_EQ(2, m0_dst(1, 0));
  EXPECT_EQ(2, m0_dst(1, 1));
}

#endif
