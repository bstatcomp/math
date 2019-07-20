#ifndef STAN_MATH_REV_SCAL_META_IS_VAR_HPP
#define STAN_MATH_REV_SCAL_META_IS_VAR_HPP

#include <stan/math/prim/scal/meta/is_var.hpp>
#include <stan/math/rev/core.hpp>

namespace stan {
/**
 * Defines a public enum named value and sets it to true(1)
 * when instantiated with the stan::math::var type.
 */
template <>
struct is_var<stan::math::var> {
  enum { value = true };
};

template <typename T>
struct is_vari : std::false_type {};

template <>
struct is_vari<stan::math::vari> : std::true_type {};

}  // namespace stan
#endif
