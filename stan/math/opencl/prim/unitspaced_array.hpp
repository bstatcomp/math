#ifndef STAN_MATH_OPENCL_PRIM_UNITSPACED_ARRAY_HPP
#define STAN_MATH_OPENCL_PRIM_UNITSPACED_ARRAY_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Return an array of integers in an ordered sequence.
 *
 * This produces an array from low to high (included).
 *
 * @param low smallest integer
 * @param high largest integer
 * @return An array of size (high - low + 1) with elements linearly spaced
 * between low and high.
 * @throw std::domain_error if high is less than low.
 */
template <typename T_x,
          require_matrix_cl_t<T_x>* = nullptr>
inline auto unitspaced_array(int low, int high) {
  check_greater_or_equal("unitspaced_array(OpenCL)", "high", high, low);
  return row_index(high-low+1,1)+low;
}
}  // namespace math
}  // namespace stan
#endif
#endif
