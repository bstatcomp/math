#ifndef STAN_MATH_GPU_TRIANGULAR_TRANSPOSE_HPP
#define STAN_MATH_GPU_TRIANGULAR_TRANSPOSE_HPP
#ifdef STAN_OPENCL

#include <stan/math/gpu/opencl_context.hpp>
#include <stan/math/gpu/constants.hpp>
#include <stan/math/gpu/kernels/triangular_transpose.hpp>
#include <stan/math/gpu/event_utils.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/gpu/matrix_gpu.hpp>

#include <CL/cl.hpp>

namespace stan {
namespace math {

/**
 * Copies a lower/upper triangular of a matrix to it's upper/lower.
 *
 * @tparam triangular_map Specifies if the copy is
 * lower-to-upper or upper-to-lower triangular. The value
 * must be of type TriangularMap
 *
 * @throw <code>std::invalid_argument</code> if the matrix is not square.
 *
 */
template <TriangularMapGPU triangular_map = TriangularMapGPU::LowerToUpper>
void matrix_gpu::triangular_transpose() {
  if (size() == 0 || size() == 1) {
    return;
  }
  check_size_match("triangular_transpose (GPU)",
                   "Expecting a square matrix; rows of ", "A", rows(),
                   "columns of ", "A", cols());

  cl::CommandQueue cmdQueue = opencl_context.queue();
  try {
    cl::Event triangular_event = opencl_kernels::triangular_transpose(this->events(),
          cl::NDRange(this->rows(), this->cols()), this->buffer(), this->rows(),
          this->cols(), triangular_map);
    this->events(triangular_event);
  } catch (const cl::Error& e) {
    check_opencl_error("triangular_transpose", e);
  }
}

}  // namespace math
}  // namespace stan

#endif
#endif
