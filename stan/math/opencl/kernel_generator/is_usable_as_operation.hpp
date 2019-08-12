#ifndef STAN_MATH_OPENCL_IS_USABLE_AS_OPERATION
#define STAN_MATH_OPENCL_IS_USABLE_AS_OPERATION
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <type_traits>

namespace stan {
namespace math {

template<typename T>
struct is_usable_as_operation{
  enum{ value = std::is_base_of<operation_base,T>::value};
};

template<typename T>
struct is_usable_as_operation<matrix_cl<T>> : std::true_type{};

template<typename T>
using enable_if_is_usable_as_operation = typename std::enable_if<is_usable_as_operation<T>::value>::type;

}
}


#endif
#endif
