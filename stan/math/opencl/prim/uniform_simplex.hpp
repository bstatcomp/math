#ifndef STAN_MATH_OPENCL_PRIM_UNIFORM_SIMPLEX_HPP
#define STAN_MATH_OPENCL_PRIM_UNIFORM_SIMPLEX_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Return a uniform simplex of size K
 *
 * @param K size of the simplex
 * @return A vector of size K with all elements initialised to 1 / K,
 * so that their sum is equal to 1.
 * @throw std::domain_error if K is not positive.
 */
template <typename T_x, require_matrix_cl_t<T_x>* = nullptr>
inline auto uniform_simplex(int K) {
  check_positive("uniform_simplex", "size", K);
  return constant(1. / K, K, 1);
}
}  // namespace math
}  // namespace stan
#endif
#endif
