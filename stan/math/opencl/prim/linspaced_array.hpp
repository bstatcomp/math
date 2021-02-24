#ifndef STAN_MATH_OPENCL_PRIM_LINSPACED_ARRAY_HPP
#define STAN_MATH_OPENCL_PRIM_LINSPACED_ARRAY_HPP
#ifdef STAN_OPENCL

#include <stan/math/prim/err.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Return a vector of linearly spaced elements.
 *
 * This produces a vector from low to high (inclusive) with elements spaced
 * as (high - low) / (K - 1). For K=1, the array will contain the high value;
 * for K=0 it returns an empty array.
 *
 * @param K size of the array
 * @param low smallest value
 * @param high largest value
 * @return An array of size K with elements linearly spaced between
 * low and high.
 * @throw std::domain_error if K is negative, if low is nan or infinite,
 * if high is nan or infinite, or if high is less than low.
 */
template <typename T_x,
          require_matrix_cl_t<T_x>* = nullptr>
inline auto linspaced_array(int K, double low, double high) {
  static const char* function = "linspaced_array (OpenCL)";
  check_nonnegative(function, "size", K);
  check_finite(function, "low", low);
  check_finite(function, "high", high);
  return select(K<=1, high, row_index(K,1) * ((high-low)/(K-1)) + low);
}
}  // namespace math
}  // namespace stan
#endif
#endif
