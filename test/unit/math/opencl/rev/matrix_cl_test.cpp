#ifdef STAN_OPENCL
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

TEST(MathMatrixCL, matrix_cl_var_creation) {
  using stan::math::vector_v;
  using stan::math::matrix_v;
  using stan::math::matrix_cl;
  using stan::math::matrix_d;
  using stan::math::var;
  vector_v vec_1;
  matrix_v mat_1;
  matrix_v mat_2;

  vec_1.resize(3);
  mat_1.resize(2, 3);
  mat_2.resize(3, 3);
  vec_1 << 1, 2, 3;
  mat_1 << 1, 2, 3, 1, 2, 3;
  mat_2 << 1, 2, 3, 1, 2, 3, 1, 2, 3;

  //EXPECT_NO_THROW(matrix_cl<var> A(1, 1));
  //EXPECT_NO_THROW(matrix_cl<var> mat_cl_1(mat_1));
  //EXPECT_NO_THROW(matrix_cl<var> mat_cl_2(mat_2));
  //EXPECT_NO_THROW(matrix_cl<var> vec_cl_2(vec_1));
  matrix_cl<var> mat_cl_1(mat_1);
  matrix_cl<double> mat_cl_1d(mat_1.val().eval());
  matrix_d mat_1d_val = stan::math::from_matrix_cl(mat_cl_1d);
  std::cout << "mat_1_val \n" << mat_1d_val << "\n";

  //matrix_cl<var> mat_cl_2(mat_2);
  //matrix_cl<var> vec_cl_2(vec_1);

  matrix_d mat_1_val = stan::math::from_matrix_cl(mat_cl_1.val());
  std::cout << "mat_1_val \n" << mat_1_val << "\n";
  matrix_d mat_1_adj = stan::math::from_matrix_cl(mat_cl_1.adj());
  std::cout << "mat_1_adj \n" << mat_1_adj << "\n";
}

#endif
