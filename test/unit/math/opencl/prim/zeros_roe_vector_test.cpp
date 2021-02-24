#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>

auto zeros_row_vector_functorCPU
    = [](int n) { return stan::math::zeros_row_vector(n); };
auto zeros_row_vector_functorCL = [](int n) {
  return stan::math::zeros_row_vector<stan::math::matrix_cl<double>>(n);
};

TEST(OpenCLZerosRowVector, scalar_prim_rev_values_small) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      zeros_row_vector_functorCPU, zeros_row_vector_functorCL, 7);
}

TEST(OpenCLZerosRowVector, scalar_prim_rev_size_0) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      zeros_row_vector_functorCPU, zeros_row_vector_functorCL, 0);
}

TEST(OpenCLZerosRowVector, scalar_prim_rev_values_large) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      zeros_row_vector_functorCPU, zeros_row_vector_functorCL, 79);
}

#endif
