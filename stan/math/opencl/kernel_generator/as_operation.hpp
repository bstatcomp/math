#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_AS_OPERATION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_AS_OPERATION_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/opencl/kernel_generator/load.hpp>
#include <stan/math/opencl/kernel_generator/constant.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <type_traits>

namespace stan {
namespace math {

template<typename T, typename Cond = std::enable_if_t<std::is_base_of<operation_base, T>::value>>
const T as_operation(const T& a){ //TODO: return ref, separate type deduction
  return a;
}

template<typename T, typename Cond = std::enable_if_t<std::is_arithmetic<T>::value>>
const constant__<T> as_operation(const T a){
  return constant__<T>(a);
}

template<typename T>
const load__<T> as_operation(const matrix_cl<T>& a){
  return load__<T>(a);
}

}
}

#endif
#endif
