#ifndef STAN_MATH_OPENCL_DIAG_MATRIX_HPP
#define STAN_MATH_OPENCL_DIAG_MATRIX_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/kernels/diag_matrix.hpp>
#include <stan/math/opencl/err/check_vector.hpp>

namespace stan {
namespace math {
template <typename T, typename = enable_if_arithmetic<T>>
inline matrix_cl<T> diag_matrix(const matrix_cl<T>& V) {
  check_vector("diag_matrix","V",V);
  int size = V.size();
  matrix_cl<T> res(size, size, matrix_cl_view::Diagonal);
  opencl_kernels::diag_matrix(cl::NDRange(size), res, V, size);
  return res;
}
}
}

#endif
#endif
