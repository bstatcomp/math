#ifndef STAN_MATH_PRIM_FUN_ONES_ARRAY_HPP
#define STAN_MATH_PRIM_FUN_ONES_ARRAY_HPP

#include <stan/math/prim/err.hpp>
#include <stan/math/prim/meta.hpp>
#include <vector>

namespace stan {
namespace math {

/**
 * Return an array of ones.
 *
 * @param K size of the array
 * @return An array of size K with all elements initialised to 1.
 * @throw std::domain_error if K is negative.
 */
template<typename T = std::vector<double>, require_std_vector_t<T>* = nullptr>
inline std::vector<double> ones_array(int K) {
  check_nonnegative("ones_array", "size", K);
  return std::vector<double>(K, 1);
}

}  // namespace math
}  // namespace stan

#endif
