#ifndef STAN_MATH_OPENCL_PRIM_LINSPACED_INT_ARRAY_HPP
#define STAN_MATH_OPENCL_PRIM_LINSPACED_INT_ARRAY_HPP
#ifdef STAN_OPENCL

#include <stan/math/prim/err.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Return a vector of linearly spaced integers.
 *
 * This produces an array from `low` to `high` (inclusive). If `high - low` is
 * greater or equal to `K - 1`, then the integers are evenly spaced. If it is
 * not possible to get from `low` to `high` with a multiple of an integer,
 * `high` is lowered until this is possible.
 *
 * If `K - 1` is greater than `high - low`, then integers are repeated. For
 * instance, `low, low, low + 1, low + 1, ...`. `high` is lowered until `K - 1`
 * is a multiple of `high - low`
 *
 * For `K = 1`, the array will contain the `high` value. For `K = 0` it returns
 * an empty array.
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
inline auto linspaced_int_array(int K, int low, int high) {
  static const char* function = "linspaced_int_array (OpenCL)";
  check_nonnegative(function, "size", K);
  check_finite(function, "low", low);
  check_finite(function, "high", high);
  bool size_is_1 = K<=1;
  int step = 0;
  if(!size_is_1){
    step = (high-low)/(K-1);
  }
  return select(size_is_1, high, row_index(K,1) * step + low);
}
}  // namespace math
}  // namespace stan
#endif
#endif
