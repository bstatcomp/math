#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_AS_OPERATION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_AS_OPERATION_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/opencl/kernel_generator/load.hpp>
#include <stan/math/opencl/kernel_generator/scalar.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <type_traits>

namespace stan {
namespace math {

template<typename T, typename = std::enable_if_t<std::is_base_of<operation_base, std::remove_reference_t<T>>::value>>
T as_operation(T&& a){ //TODO: return ref, separate type deduction
  return a;
}

template<typename T, typename = enable_if_arithmetic<T>>
scalar__<T> as_operation(const T a){
  return scalar__<T>(a);
}

template<typename T, typename = std::enable_if_t<std::is_base_of<matrix_cl<typename std::remove_reference_t<T>::type>,typename std::remove_reference_t<T>>::value>>
load__<T> as_operation(T&& a){
  return load__<T>(std::forward<T>(a));
}

}
}

#endif
#endif
