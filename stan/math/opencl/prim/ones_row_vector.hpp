#ifndef STAN_MATH_OPENCL_PRIM_ONES_ROW_VECTOR_HPP
#define STAN_MATH_OPENCL_PRIM_ONES_ROW_VECTOR_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Return a row vector of ones
 *
 * @param K size of the row vector
 * @return A row vector of size K with all elements initialised to 1.
 * @throw std::domain_error if K is negative.
 */
template <typename T_x,
          require_matrix_cl_t<T_x>* = nullptr>
inline auto ones_row_vector(int K) {
  check_nonnegative("ones_row_vector(OpenCL)", "size", K);
  return constant(1.0,1, K);
}
}  // namespace math
}  // namespace stan
#endif
#endif
