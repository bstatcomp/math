#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_AS_OPERATION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_AS_OPERATION_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/load.hpp>
#include <stan/math/opencl/matrix_cl.hpp>

namespace stan {
namespace math {

template<typename T>
const T& as_operation(const T& a){
  return a;
}

template<typename T>
const load__<T> as_operation(const matrix_cl<T>& a){
  return load__<T>(a);
}

}
}

#endif
#endif
