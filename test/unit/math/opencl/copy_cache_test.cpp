#ifdef STAN_OPENCL
#define STAN_OPENCL_CACHE
#include <stan/math/prim/mat.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/copy.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>

TEST(MathMatrixOpenCL, matrix_cl_copy_cache) {
  auto m = stan::math::matrix_d::Random(100, 100).eval();
  stan::math::matrix_cl d11(100, 100);
  stan::math::matrix_cl d12(100, 100);
  d11 = stan::math::to_matrix_cl(m);
  d12 = stan::math::to_matrix_cl(m);
  ASSERT_FALSE(m.opencl_buffer_() == NULL);
}

TEST(MathMatrixOpenCL, matrix_cl_var_copy) {
  Eigen::Matrix<stan::math::var, Eigen::Dynamic, Eigen::Dynamic> m(5, 5);
  double pos_ = 1.1;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      m(i, j) = pos_++;

  stan::math::matrix_cl d1_cl(5, 5);
  d1_cl = stan::math::to_matrix_cl(m);
  EXPECT_TRUE(m.opencl_buffer_() == NULL);
  stan::math::matrix_d d1_cpu_return(5, 5);
  d1_cpu_return = stan::math::from_matrix_cl(d1_cl);
  EXPECT_EQ(1.1, d1_cpu_return(0, 0));
  EXPECT_EQ(6.1, d1_cpu_return(1, 0));
  EXPECT_EQ(11.1, d1_cpu_return(2, 0));
  EXPECT_EQ(16.1, d1_cpu_return(3, 0));
  EXPECT_EQ(21.1, d1_cpu_return(4, 0));
}

TEST(MathMatrixOpenCL, matrix_cl_copy) {
  stan::math::vector_d d1;
  stan::math::vector_d d1_a;
  stan::math::vector_d d1_b;
  stan::math::matrix_d d2;
  stan::math::matrix_d d2_a;
  stan::math::matrix_d d2_b;
  stan::math::matrix_d d0;
  d1.resize(3);
  d1_a.resize(3);
  d1_b.resize(3);
  d1 << 1, 2, 3;
  d2.resize(2, 3);
  d2_a.resize(2, 3);
  d2_b.resize(2, 3);
  d2 << 1, 2, 3, 4, 5, 6;
  // vector
  stan::math::matrix_cl d11(3, 1);
  stan::math::matrix_cl d111(3, 1);
  EXPECT_NO_THROW(d11 = stan::math::to_matrix_cl(d1));
  EXPECT_NO_THROW(d111 = stan::math::copy_cl(d11));
  EXPECT_NO_THROW(d1_a = stan::math::from_matrix_cl(d11));
  EXPECT_NO_THROW(d1_b = stan::math::from_matrix_cl(d111));
  EXPECT_EQ(1, d1_a(0));
  EXPECT_EQ(2, d1_a(1));
  EXPECT_EQ(3, d1_a(2));
  EXPECT_EQ(1, d1_b(0));
  EXPECT_EQ(2, d1_b(1));
  EXPECT_EQ(3, d1_b(2));
  // matrix
  stan::math::matrix_cl d00;
  stan::math::matrix_cl d000;
  stan::math::matrix_cl d22(2, 3);
  stan::math::matrix_cl d222(2, 3);
  EXPECT_NO_THROW(d22 = stan::math::to_matrix_cl(d2));
  EXPECT_NO_THROW(d222 = stan::math::copy_cl(d22));
  EXPECT_NO_THROW(d2_a = stan::math::from_matrix_cl(d22));
  EXPECT_NO_THROW(d2_b = stan::math::from_matrix_cl(d222));
  EXPECT_EQ(1, d2_a(0, 0));
  EXPECT_EQ(2, d2_a(0, 1));
  EXPECT_EQ(3, d2_a(0, 2));
  EXPECT_EQ(4, d2_a(1, 0));
  EXPECT_EQ(5, d2_a(1, 1));
  EXPECT_EQ(6, d2_a(1, 2));
  EXPECT_EQ(1, d2_b(0, 0));
  EXPECT_EQ(2, d2_b(0, 1));
  EXPECT_EQ(3, d2_b(0, 2));
  EXPECT_EQ(4, d2_b(1, 0));
  EXPECT_EQ(5, d2_b(1, 1));
  EXPECT_EQ(6, d2_b(1, 2));

  // zero sized copy
  // cpu to gpu
  EXPECT_NO_THROW(d00 = stan::math::to_matrix_cl(d0));
  // gpu to cpu
  EXPECT_NO_THROW(d0 = stan::math::from_matrix_cl(d00));
  // gpu to gpu
  EXPECT_NO_THROW(d000 = stan::math::copy_cl(d00));
}

TEST(MathMatrixOpenCL, barebone_buffer_copy) {
  // a barebone OpenCL example of copying
  // a vector of doubles to the GPU and back
  size_t size = 512;
  std::vector<double> cpu_buffer(size);
  for (unsigned int i = 0; i < size; i++) {
    cpu_buffer[i] = i * 1.0;
  }
  std::vector<double> cpu_dst_buffer(size);
  // retrieve the command queue
  cl::CommandQueue queue = stan::math::opencl_context.queue();
  // retrieve the context
  cl::Context& ctx = stan::math::opencl_context.context();
  // create the gpu buffer of the same size
  cl::Buffer gpu_buffer
      = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * size);

  // write the cpu_buffer to the GPU (gpu_buffer)
  queue.enqueueWriteBuffer(gpu_buffer, CL_TRUE, 0, sizeof(double) * size,
                           &cpu_buffer[0]);

  // write the gpu buffer back to the cpu_dst_buffer
  queue.enqueueReadBuffer(gpu_buffer, CL_TRUE, 0, sizeof(double) * size,
                          &cpu_dst_buffer[0]);

  for (unsigned int i = 0; i < size; i++) {
    EXPECT_EQ(i * 1.0, cpu_dst_buffer[i]);
  }
}
#undef STAN_OPENCL
#endif
