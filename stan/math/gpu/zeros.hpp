#ifndef STAN_MATH_GPU_ZEROS_HPP
#define STAN_MATH_GPU_ZEROS_HPP
#ifdef STAN_OPENCL

#include <stan/math/gpu/opencl_context.hpp>
#include <stan/math/gpu/constants.hpp>
#include <stan/math/gpu/kernels/zeros.hpp>
#include <stan/math/gpu/event_utils.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/gpu/matrix_gpu.hpp>

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
template <TriangularViewGPU triangular_view = TriangularViewGPU::Entire>
void matrix_gpu::zeros() {
  if (size() == 0)
    return;
  cl::CommandQueue cmdQueue = opencl_context.queue();
  try {
    cl::Event zero_event = opencl_kernels::zeros(this->events(),
       cl::NDRange(this->rows(), this->cols()), this->buffer(), this->rows(),
       this->cols(), triangular_view);
    this->events(zero_event);
  } catch (const cl::Error& e) {
    check_opencl_error("zeros", e);
  }
}

}  // namespace math
}  // namespace stan

#endif
#endif
