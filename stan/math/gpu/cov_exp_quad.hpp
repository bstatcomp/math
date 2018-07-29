#ifndef STAN_MATH_GPU_COV_EXP_QUAD_HPP
#define STAN_MATH_GPU_COV_EXP_QUAD_HPP
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
inline void cov_exp_quad(matrix_gpu x, matrix_gpu pos, matrix_gpu cnst,
    matrix_gpu &dist, matrix_gpu &temp) {
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

/**
 * Returns the identity matrix stored on the GPU
 *
 * @param rows_cols the number of rows and columns
 *
 * @return the identity matrix
 *
 */
inline void cov_exp_quad2(matrix_gpu x, matrix_gpu cnst, matrix_gpu& res1) {
  cl::Kernel kernel = opencl_context.get_kernel("cov_exp_quad2");
  cl::CommandQueue cmdQueue = opencl_context.queue();

  try {
    kernel.setArg(0, x.buffer());
    kernel.setArg(1, cnst.buffer());
    kernel.setArg(2, res1.buffer());
    kernel.setArg(3, x.size());
    cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                  cl::NDRange(x.size(), x.size()),
                                  cl::NullRange, NULL, NULL);
  } catch (const cl::Error& e) {
    check_opencl_error("cov_exp_quad2", e);
  }
}

/**
 * Returns the identity matrix stored on the GPU
 *
 * @param rows_cols the number of rows and columns
 *
 * @return the identity matrix
 *
 */
inline void cov_exp_quad3(matrix_gpu x1, matrix_gpu x2,
    matrix_gpu cnst, matrix_gpu& res1) {
  cl::Kernel kernel = opencl_context.get_kernel("cov_exp_quad3");
  cl::CommandQueue cmdQueue = opencl_context.queue();

  try {
    kernel.setArg(0, x1.buffer());
    kernel.setArg(1, x2.buffer());
    kernel.setArg(2, cnst.buffer());
    kernel.setArg(3, res1.buffer());
    kernel.setArg(4, x1.size());
    cmdQueue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                  cl::NDRange(x1.size(), x1.size()),
                                  cl::NullRange, NULL, NULL);
  } catch (const cl::Error& e) {
    check_opencl_error("cov_exp_quad2", e);
  }
}
}  // namespace math
}  // namespace stan

#endif
#endif
