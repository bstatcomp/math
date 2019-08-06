#ifdef STAN_OPENCL
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat.hpp>
#include <stan/math/opencl/rev/copy.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/rev/sub_block.hpp>
#include <stan/math/opencl/copy.hpp>
#include <gtest/gtest.h>
#include <algorithm>

TEST(MathMatrixCL, sub_block_pass) {
  using stan::math::var;
  stan::math::matrix_v d1;
  stan::math::matrix_v d2;
  d1.resize(3, 3);
  d2.resize(4, 4);

  d1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  d2 << 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1;

  stan::math::matrix_cl<var> d11(d1);
  stan::math::matrix_cl<var> d22(d2);
  d22.sub_block(d11, 0, 0, 0, 0, 2, 2);
  d2 = stan::math::from_matrix_cl(d22);
  EXPECT_EQ(1, d2.val().eval()(0, 0));
  EXPECT_EQ(2, d2.val().eval()(0, 1));
  EXPECT_EQ(4, d2.val().eval()(1, 0));
  EXPECT_EQ(5, d2.val().eval()(1, 1));
}

TEST(MathMatrixCL, sub_block_exception) {
  using stan::math::var;

  stan::math::matrix_v d1(3, 3);
  stan::math::matrix_v d2(4, 4);
  d1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  d2 << 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1;

  stan::math::matrix_cl<var> d11(d1);
  stan::math::matrix_cl<var> d22(d2);
  EXPECT_THROW(d22.sub_block(d11, 1, 1, 0, 0, 4, 4), std::domain_error);
  EXPECT_THROW(d22.sub_block(d11, 4, 4, 0, 0, 2, 2), std::domain_error);

}

TEST(MathMatrixCL, sub_block_triangular) {
  using stan::math::var;

  stan::math::matrix_v a = stan::math::matrix_v::Zero(3, 3);
  stan::math::matrix_v b = stan::math::matrix_v::Ones(3, 3);
  stan::math::matrix_cl<var> a_cl(3, 3);
  stan::math::matrix_cl<var> b_cl(b);
  Eigen::MatrixXd c;

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  a_cl.sub_block(b_cl, 0, 1, 0, 1, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Lower);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  a_cl.sub_block(b_cl, 1, 0, 1, 1, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Entire);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 1);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 1);
  EXPECT_EQ(c(2, 2), 1);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Upper);
  a_cl.sub_block(b_cl, 0, 0, 1, 0, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Lower);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 1);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 1);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Upper);
  a_cl.sub_block(b_cl, 0, 0, 0, 0, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Entire);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 1);
  EXPECT_EQ(c(0, 1), 1);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Upper);
  a_cl.sub_block(b_cl, 1, 0, 1, 0, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Diagonal);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Lower);
  b_cl.view(stan::math::matrix_cl_view::Upper);
  a_cl.sub_block(b_cl, 1, 0, 1, 0, 2, 3);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Upper);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 1);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 1);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Upper);
  a_cl.sub_block(b_cl, 1, 0, 1, 0, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Upper);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Upper);
  a_cl.sub_block(b_cl, 0, 1, 1, 1, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Entire);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 1);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 1);
  EXPECT_EQ(c(2, 2), 1);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  a_cl.sub_block(b_cl, 0, 0, 0, 1, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Upper);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 1);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 1);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  a_cl.sub_block(b_cl, 0, 0, 0, 0, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Entire);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 1);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 1);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  a_cl.sub_block(b_cl, 0, 1, 0, 1, 2, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Diagonal);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 0);
  EXPECT_EQ(c(2, 2), 0);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  a_cl.sub_block(b_cl, 0, 1, 0, 1, 3, 2);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Lower);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 0);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 0);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 0);
  EXPECT_EQ(c(2, 1), 1);
  EXPECT_EQ(c(2, 2), 1);

  a_cl = stan::math::to_matrix_cl(a);
  a_cl.view(stan::math::matrix_cl_view::Upper);
  b_cl.view(stan::math::matrix_cl_view::Lower);
  a_cl.sub_block(b_cl, 0, 0, 0, 0, 3, 3);
  EXPECT_EQ(a_cl.view(), stan::math::matrix_cl_view::Lower);
  c = stan::math::from_matrix_cl(a_cl.val());
  EXPECT_EQ(c(0, 0), 1);
  EXPECT_EQ(c(0, 1), 0);
  EXPECT_EQ(c(0, 2), 0);
  EXPECT_EQ(c(1, 0), 1);
  EXPECT_EQ(c(1, 1), 1);
  EXPECT_EQ(c(1, 2), 0);
  EXPECT_EQ(c(2, 0), 1);
  EXPECT_EQ(c(2, 1), 1);
  EXPECT_EQ(c(2, 2), 1);
}
#endif
