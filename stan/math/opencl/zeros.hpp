#ifndef STAN_MATH_GPU_ZEROS_HPP
#define STAN_MATH_GPU_ZEROS_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/constants.hpp>
#include <stan/math/opencl/kernels/zeros.hpp>
#include <stan/math/opencl/event_utils.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/opencl/matrix_cl.hpp>

#include <CL/cl.hpp>

namespace stan {
namespace math {

/**
 * Stores zeros in the matrix on the GPU.
 * Supports writing zeroes to the lower and upper triangular or
 * the whole matrix.
 *
 * @tparam triangular_view Specifies if zeros are assigned to
 * the entire matrix, lower triangular or upper triangular. The
 * value must be of type TriangularViewGPU
 */
template <TriangularViewCL triangular_view = TriangularViewCL::Entire>
inline void matrix_cl::zeros() {
  if (size() == 0)
    return;
  cl::CommandQueue cmdQueue = opencl_context.queue();
  try {
    std::vector<cl::Event> this_events = this->events();
    auto zeros_cl = opencl_kernels::zeros(cl::NDRange(this->rows(), this->cols()), this_events);
    cl::Event zero_event = zeros_cl(this->buffer(), this->rows(), this->cols(), triangular_view);
    this->events(zero_event);
  } catch (const cl::Error& e) {
    check_opencl_error("zeros", e);
  }
}

}  // namespace math
}  // namespace stan

#endif
#endif
