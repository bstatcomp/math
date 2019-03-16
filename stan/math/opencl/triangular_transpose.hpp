#ifndef STAN_MATH_GPU_TRIANGULAR_TRANSPOSE_HPP
#define STAN_MATH_GPU_TRIANGULAR_TRANSPOSE_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/constants.hpp>
#include <stan/math/opencl/kernels/triangular_transpose.hpp>
#include <stan/math/opencl/event_utils.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/opencl/matrix_cl.hpp>

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
template <TriangularMapCL triangular_map = TriangularMapCL::LowerToUpper>
inline void matrix_cl::triangular_transpose() {
  if (size() == 0 || size() == 1) {
    return;
  }
  check_size_match("triangular_transpose (GPU)",
                   "Expecting a square matrix; rows of ", "A", rows(),
                   "columns of ", "A", cols());

  cl::CommandQueue cmdQueue = opencl_context.queue();
  try {
    auto tri_trans = opencl_kernels::triangular_transpose(cl::NDRange(this->rows(), this->cols()),
     this->events());
    cl::Event triangular_event = tri_trans(this->buffer(), this->rows(),
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
