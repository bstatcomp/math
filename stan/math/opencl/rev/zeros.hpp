#ifndef STAN_MATH_OPENCL_ZEROS_HPP
#define STAN_MATH_OPENCL_ZEROS_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/constants.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/zeros.hpp>

#include <CL/cl.hpp>

namespace stan {
namespace math {

/**
 * Stores zeros in the matrix on the OpenCL device.
 * Supports writing zeroes to the lower and upper triangular or
 * the whole matrix.
 *
 * @tparam triangular_view Specifies if zeros are assigned to
 * the entire matrix, lower triangular or upper triangular. The
 * value must be of type TriangularViewCL
 */
template <typename T, enable_if_var_or_vari<T>>
template <TriangularViewCL triangular_view = TriangularViewCL::Entire, typename = enable_if_var_or_vari<T>>
inline void matrix_cl<T, enable_if_var_or_vari<T>>::zeros() try {
  this->val().zeros<triangular_view>()
  this->adj().zeros<triangular_view>()
} catch (const cl::Error& e) {
  check_opencl_error("zeros", e);
}

}  // namespace math
}  // namespace stan

#endif
#endif
