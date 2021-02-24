#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>

auto linspaced_int_array_functorCPU = [](int N, int low, int high) {
  return stan::math::linspaced_int_array(N, low, high);
};
auto linspaced_int_array_functorCL = [](int N, int low, int high) {
  return stan::math::linspaced_int_array<stan::math::matrix_cl<int>>(N, low,
                                                                     high);
};

TEST(OpenCLLinspacedIntArray, scalar_prim_rev_values_small) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_int_array_functorCPU, linspaced_int_array_functorCL, 7, 3, 15);
}

TEST(OpenCLLinspacedIntArray, scalar_prim_rev_size_1) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_int_array_functorCPU, linspaced_int_array_functorCL, 1, 1, 4);
}

TEST(OpenCLLinspacedIntArray, scalar_prim_rev_size_0) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_int_array_functorCPU, linspaced_int_array_functorCL, 0, 1, 4);
}

TEST(OpenCLLinspacedIntArray, scalar_prim_rev_values_large) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      linspaced_int_array_functorCPU, linspaced_int_array_functorCL, 79, 23,
      123);
}

#endif
