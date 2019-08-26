#ifndef STAN_MATH_PRIM_ARR_META_IS_CONSTANT_HPP
#define STAN_MATH_PRIM_ARR_META_IS_CONSTANT_HPP

#include <stan/math/prim/scal/meta/is_constant.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <type_traits>
#include <vector>

namespace stan {
/**
 * Defines a public enum named value and sets it to true
 * if the type of the elements in the provided std::vector
 * is constant, false otherwise. This is used in
 * the is_constant_all metaprogram.
 * @tparam type of the elements in the std::vector
 */
template <typename T>
struct is_constant<T, std::enable_if_t<is_std_vector<std::decay_t<T>>::value>>
    : std::integral_constant<bool, is_constant<typename T::value_type>::value> {
};

}  // namespace stan
#endif
