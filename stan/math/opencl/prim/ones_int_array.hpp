#ifndef STAN_MATH_OPENCL_PRIM_ONES_INT_ARRAY_HPP
#define STAN_MATH_OPENCL_PRIM_ONES_INT_ARRAY_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Return a vector of ones
 *
 * @param K size of the vector
 * @return A vector of size K with all elements initialised to 1.
 * @throw std::domain_error if K is negative.
 */
template <typename T_x,
          require_matrix_cl_t<T_x>* = nullptr>
inline auto ones_int_array(int K) {
  check_nonnegative("ones_int_array(OpenCL)", "size", K);
  return constant(1,K,1);
}
}  // namespace math
}  // namespace stan
#endif
#endif
