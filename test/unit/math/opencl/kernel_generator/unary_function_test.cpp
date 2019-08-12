#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/unary_function.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>

using Eigen::Dynamic;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using stan::math::matrix_cl;

#define EXPECT_MATRIX_NEAR(A, B, DELTA) \
  for (int i = 0; i < A.size(); i++)    \
    EXPECT_NEAR(A(i), B(i), DELTA);

#define TEST_FUNCTION(fun) \
TEST(MathMatrixCL, fun##_test){ \
  using stan::math::fun; \
  MatrixXd m1(3, 3); \
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9; \
\
  matrix_cl<double> m1_cl(m1);\
  auto tmp = fun(m1_cl); \
  matrix_cl<double> res_cl = tmp;\
\
  MatrixXd res = stan::math::from_matrix_cl(res_cl); \
\
  EXPECT_MATRIX_NEAR(fun(m1),res,1e-9); \
}

TEST_FUNCTION(sqrt)
//TEST_FUNCTION(cbrt)
//
//TEST_FUNCTION(exp)
//TEST_FUNCTION(exp2)
//TEST_FUNCTION(expm1)
//TEST_FUNCTION(log)
//TEST_FUNCTION(log1p)
//
//TEST_FUNCTION(sin)
//TEST_FUNCTION(sinh)
//TEST_FUNCTION(cos)
//TEST_FUNCTION(cosh)
//TEST_FUNCTION(tan)
//TEST_FUNCTION(tanh)
//
//TEST_FUNCTION(tgamma)
//TEST_FUNCTION(lgamma)
//TEST_FUNCTION(erf)
//TEST_FUNCTION(erfc)


#endif