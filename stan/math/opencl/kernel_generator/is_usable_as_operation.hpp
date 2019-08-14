#ifndef STAN_MATH_OPENCL_IS_USABLE_AS_OPERATION
#define STAN_MATH_OPENCL_IS_USABLE_AS_OPERATION
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/prim/meta.hpp>
#include <type_traits>

namespace stan {
namespace math {

template<typename T>
struct is_usable_as_operation_and_not_arithmetic{
  enum{ value = std::is_base_of<operation_base,T>::value};
};

template<typename T>
struct is_usable_as_operation_and_not_arithmetic<matrix_cl < T>> : std::true_type{};

template<typename T>
struct is_usable_as_operation_and_not_arithmetic<T&>{
    enum{ value = is_usable_as_operation_and_not_arithmetic<T>::value};
};

template<typename T>
struct is_usable_as_operation_and_not_arithmetic<T&&>{
    enum{ value = is_usable_as_operation_and_not_arithmetic<T>::value};
};

template<typename T>
struct is_usable_as_operation{
    enum{ value = is_usable_as_operation_and_not_arithmetic<T>::value || std::is_arithmetic<std::remove_reference_t<T>>::value};
};

template<typename... Types>
using enable_if_none_arithmetic_all_usable_as_operation = typename std::enable_if_t<math::conjunction<is_usable_as_operation_and_not_arithmetic < Types>...>::value>;

template<typename... Types>
using enable_if_all_usable_as_operation = typename std::enable_if_t<math::conjunction<is_usable_as_operation<Types>...>::value>;

}
}


#endif
#endif
