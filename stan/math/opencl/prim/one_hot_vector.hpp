#ifndef STAN_MATH_OPENCL_PRIM_ONE_HOT_VECTOR_HPP
#define STAN_MATH_OPENCL_PRIM_ONE_HOT_VECTOR_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Return a vector with 1 in the k-th position and zero elsewhere.
 *
 * @param K size of the vector
 * @param k position of the 1 (indexing from 1)
 * @return A vector of size K with all elements initialised to zero
 * except a 1 in the k-th position.
 * @throw std::domain_error if K is not positive, or if k is less than 1 or
 * greater than K.
 */
template <typename T_x,
          require_matrix_cl_t<T_x>* = nullptr>
inline auto one_hot_vector(int K, int k) {
  static const char* function = "one_hot_vector (OpenCL)";
  check_positive(function, "size", K);
  check_bounded(function, "k", k, 1, K);
  return select(row_index(K,1)==k-1, 1.0, 0.0);
}
}  // namespace math
}  // namespace stan
#endif
#endif
