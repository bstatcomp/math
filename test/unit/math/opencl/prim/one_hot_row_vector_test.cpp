#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>

auto one_hot_row_vector_functorCPU
    = [](int N, int n) { return stan::math::one_hot_row_vector(N,n); };
auto one_hot_row_vector_functorCL = [](int N, int n) {
  return stan::math::one_hot_row_vector<stan::math::matrix_cl<double>>(N,n);
};

TEST(OpenCLOneHotRowVector, scalar_prim_rev_values_small) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      one_hot_row_vector_functorCPU, one_hot_row_vector_functorCL, 7, 3);
}

TEST(OpenCLOneHotRowVector, scalar_prim_rev_size_1) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      one_hot_row_vector_functorCPU, one_hot_row_vector_functorCL, 1,1);
}

TEST(OpenCLOneHotRowVector, scalar_prim_rev_values_large) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      one_hot_row_vector_functorCPU, one_hot_row_vector_functorCL, 79,23);
}

#endif
