#ifndef STAN_MATH_PRIM_MAT_FUN_SQRT_HPP
#define STAN_MATH_PRIM_MAT_FUN_SQRT_HPP

#include <stan/math/prim/mat/vectorize/apply_scalar_unary.hpp>
#include <type_traits>
#include <cmath>

namespace stan {
namespace math {

/**
 * Structure to wrap sqrt() so that it can be vectorized.
 * @param x Variable.
 * @tparam T Variable type.
 * @return Square root of x.
 */
struct sqrt_fun {
  template <typename T>
  static inline T fun(const T& x) {
    using std::sqrt;
    return sqrt(x);
  }
};

/**
 * Vectorized version of sqrt().
 * @param x Container.
 * @tparam T Container type.
 * @return Square root of each value in x.
 */
template <typename T, typename = std::enable_if_t<apply_scalar_unary<sqrt_fun, T>::enabled>>
inline typename apply_scalar_unary<sqrt_fun, T>::return_t sqrt(const T& x) {
  return apply_scalar_unary<sqrt_fun, T>::apply(x);
}

}  // namespace math
}  // namespace stan

#endif
