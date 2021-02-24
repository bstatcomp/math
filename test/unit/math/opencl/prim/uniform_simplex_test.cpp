#ifdef STAN_OPENCL
#include <stan/math/opencl/rev.hpp>
#include <stan/math.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/opencl/util.hpp>
#include <test/unit/util.hpp>

auto uniform_simplex_functorCPU
    = [](int N) { return stan::math::uniform_simplex(N); };
auto uniform_simplex_functorCL = [](int N) {
  return stan::math::uniform_simplex<stan::math::matrix_cl<double>>(N);
};

TEST(OpenCLUniformSimplex, scalar_prim_rev_values_small) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      uniform_simplex_functorCPU, uniform_simplex_functorCL, 7);
}

TEST(OpenCLUniformSimplex, scalar_prim_rev_size_1) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      uniform_simplex_functorCPU, uniform_simplex_functorCL, 1);
}

TEST(OpenCLUniformSimplex, scalar_prim_rev_values_large) {
  stan::math::test::compare_cpu_opencl_prim_rev_separate(
      uniform_simplex_functorCPU, uniform_simplex_functorCL, 79);
}

#endif
