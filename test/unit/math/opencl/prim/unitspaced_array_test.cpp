#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>

auto unitspaced_array_functorCPU
    = [](int N, int n) { return stan::math::unitspaced_array(N,n); };
auto unitspaced_array_functorCL = [](int N, int n) {
  return stan::math::unitspaced_array<stan::math::matrix_cl<double>>(N,n);
};

TEST(OpenCLUnitspacedArray, scalar_prim_rev_values_small) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      unitspaced_array_functorCPU, unitspaced_array_functorCL, 3, 7);
}

TEST(OpenCLUnitspacedArray, scalar_prim_rev_size_1) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      unitspaced_array_functorCPU, unitspaced_array_functorCL, 1,1);
}

TEST(OpenCLUnitspacedArray, scalar_prim_rev_values_large) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      unitspaced_array_functorCPU, unitspaced_array_functorCL, 23,129);
}

#endif
