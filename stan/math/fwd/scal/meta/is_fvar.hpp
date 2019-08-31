#ifndef STAN_MATH_FWD_SCAL_META_IS_FVAR_HPP
#define STAN_MATH_FWD_SCAL_META_IS_FVAR_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/meta/is_fvar.hpp>

namespace stan {
/**
 * Specialization of is_fvar with a member value indicating the type is fvar.
 */
template <typename T>
struct is_fvar<stan::math::fvar<T>> : std::true_type {};

}  // namespace stan
#endif
