#ifndef STAN_MATH_PRIM_SCAL_META_ENABLE_IF_VAR_OR_VARI_HPP
#define STAN_MATH_PRIM_SCAL_META_ENABLE_IF_VAR_OR_VARI_HPP

#include <type_traits>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
namespace stan {

template <typename T>
using enable_if_var_or_vari = std::enable_if_t<is_var<T>::value>;

}  // namespace stan
#endif
