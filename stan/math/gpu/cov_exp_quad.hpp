#ifndef STAN_MATH_GPU_IDENTITY_HPP
#define STAN_MATH_GPU_IDENTITY_HPP
#ifdef STAN_OPENCL
#include <stan/math/gpu/matrix_gpu.hpp>
#include <CL/cl.hpp>

namespace stan {
namespace math {

/**
 * Returns the identity matrix stored on the GPU
 *
 * @param rows_cols the number of rows and columns
 *
 * @return the identity matrix
 *
 */
inline void cov_exp_quad(matrix_gpu x, matrix_gpu pos, matrix_gpu cnst, matrix_gpu &dist, matrix_gpu &temp) {
  cl::Kernel kernel = opencl_context.get_kernel("cov_exp_quad");
  cl::CommandQueue cmdQueue = opencl_context.queue();

  try {
    kernel.setArg(0, x.buffer());
    kernel.setArg(1, pos.buffer());
    kernel.setArg(2, cnst.buffer());
    kernel.setArg(3, dist.buffer());
    kernel.setArg(4, temp.buffer());
    kernel.setArg(5, x.size());
    cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                  cl::NDRange(x.size(), x.size()),
                                  cl::NullRange, NULL, NULL);
  } catch (const cl::Error& e) {
    check_opencl_error("cov_exp_quad", e);
  }
}
}  // namespace math
}  // namespace stan

#endif
#endif
