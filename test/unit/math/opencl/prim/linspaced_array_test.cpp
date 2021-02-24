#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>

auto linspaced_array_functorCPU = [](int N, auto low, auto high) {
  return stan::math::linspaced_array(N, stan::math::value_of(low),
                                     stan::math::value_of(high));
};
auto linspaced_array_functorCL = [](int N, auto low, auto high) {
  return stan::math::linspaced_array<stan::math::matrix_cl<double>>(
      N, stan::math::value_of(low), stan::math::value_of(high));
};

TEST(OpenCLLinspacedArray, scalar_prim_rev_values_small) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_array_functorCPU, linspaced_array_functorCL, 7, 3.5, 15.6);
}

TEST(OpenCLLinspacedArray, scalar_prim_rev_size_1) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_array_functorCPU, linspaced_array_functorCL, 1, 1.1, 1.6);
}

TEST(OpenCLLinspacedArray, scalar_prim_rev_size_0) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_array_functorCPU, linspaced_array_functorCL, 0, 1.1, 1.6);
}

TEST(OpenCLLinspacedArray, scalar_prim_rev_values_large) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_array_functorCPU, linspaced_array_functorCL, 79, 23.9, 123.456);
}

#endif
