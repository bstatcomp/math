#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>

auto zeros_array_functorCPU
    = [](int n) { return stan::math::zeros_array(n); };
auto zeros_array_functorCL = [](int n) {
  return stan::math::zeros_array<stan::math::matrix_cl<double>>(n);
};

TEST(OpenCLZerosArray, scalar_prim_rev_values_small) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      zeros_array_functorCPU, zeros_array_functorCL, 7);
}

TEST(OpenCLZerosArray, scalar_prim_rev_size_0) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      zeros_array_functorCPU, zeros_array_functorCL, 0);
}

TEST(OpenCLZerosArray, scalar_prim_rev_values_large) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      zeros_array_functorCPU, zeros_array_functorCL, 79);
}

#endif
