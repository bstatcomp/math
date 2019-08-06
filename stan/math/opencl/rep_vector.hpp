
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/kernels/constant.hpp>

#ifndef STAN_MATH_OPENCL_REP_VECTOR_HPP
#define STAN_MATH_OPENCL_REP_VECTOR_HPP
#ifdef STAN_OPENCL

namespace stan {
namespace math {
template <typename T, typename = enable_if_arithmetic<T>>
inline matrix_cl<T> rep_vector_cl(double val, int n) {
  try {
    matrix_cl<T> res(n, 1);
    opencl_kernels::constants(cl::NDRange(n, 1), res, val,
                              n, 1, matrix_cl_view::Entire);
    return res;
  } catch (const cl::Error& e) {
    check_opencl_error("rep_vector_cl", e);
  }
}
}
}

#endif
#endif //STAN_MATH_OPENCL_REP_VECTOR_HPP
