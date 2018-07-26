#ifndef STAN_MATH_PRIM_MAT_FUN_CHOLESKY_DECOMPOSE_GPU_HPP
#define STAN_MATH_PRIM_MAT_FUN_CHOLESKY_DECOMPOSE_GPU_HPP
#ifdef STAN_OPENCL
#include <stan/math/gpu/opencl_context.hpp>
#include <stan/math/gpu/matrix_gpu.hpp>
#include <stan/math/gpu/subtract.hpp>
#include <stan/math/gpu/transpose.hpp>
#include <stan/math/gpu/multiply.hpp>
#include <stan/math/gpu/multiply_self_transpose.hpp>
#include <stan/math/gpu/inverse_gpu.hpp>
#include <stan/math/gpu/err/check_diagonal_zeros.hpp>
#include <stan/math/gpu/err/check_nan.hpp>
#include <stan/math/gpu/err/check_symmetric.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <iostream>
#include <string>
#include <map>
#include <fstream>

/*   @file stanmathcl/matrix_inverse.hpp
*    @brief matrix_inverse  -  functions for matrix inversion:
*     lower triangular,  upper triangular,  regular,  ...
*/

// CURRENTLY ONLY SUPPORTS LOWER TRIANGULAR
namespace stan {
  namespace math {
    /**
     * Return the lower-triangular Cholesky factor (i.e., matrix
     * square root) of the specified square, symmetric matrix.  
     * The return value \f$L\f$ will be a lower-traingular matrix such that the
     * original matrix \f$A\f$ is given by
     * <p>\f$A = L \times L^T\f$.
     * The Cholesky decomposition is computed on the GPU. The
     * input matrix is transfered to the GPU and the resulting
     * lower-triangular matrix is then copied from the GPU.
     * 
     * @param A Symmetrix matrix on the GPU.
     * @param block size of the block for each step
     * @return Square root of matrix on the GPU.
     * @throw std::domain_error if m is not
     *  positive definite (if m has more than 0 elements)
     */    
    inline matrix_gpu cholesky_decompose_gpu(matrix_gpu& A, int block) {
      cl::Kernel kernel_chol_block
         = opencl_context.get_kernel("cholesky_block");
      cl::CommandQueue cmd_queue = opencl_context.queue();
      // Will be managed by the library core system
      int offset = 0;
      matrix_gpu V(block, block);
      matrix_gpu D(block, block);
      while ((offset + block) < (A.rows())) {
        matrix_gpu L(A.rows()-offset-block, block);
        matrix_gpu Mid(A.rows()-offset-block, A.rows()-offset-block);
        matrix_gpu Mid_temp(A.rows()-offset-block, A.rows()-offset-block);

        D.sub_block(A, offset, offset, 0, 0, block, block);
        V.zeros();
        try {
          kernel_chol_block.setArg(0, V.buffer());
          kernel_chol_block.setArg(1, D.buffer());
          kernel_chol_block.setArg(2, block);
          cmd_queue.enqueueNDRangeKernel(kernel_chol_block,
           cl::NullRange, cl::NDRange(block), cl::NDRange(block));
        } catch (const cl::Error& e) {
          check_opencl_error("cholesky_decompose", e);
        }
        copy(D, V);
        A.sub_block(V, 0, 0, offset, offset, block, block);

        V = lower_triangular_inverse(D);
        
        L.sub_block(A, (offset+block), offset, 0, 0,
          (A.rows()-offset-block) , block);
        V = transpose(V);
        L = multiply(L, V);
        A.sub_block(L, 0, 0, (offset+block), offset,
          (A.rows()-offset-block) , block);

        Mid_temp.sub_block(A, (offset+block), (offset+block),
          0, 0, (A.rows()-offset-block), (A.rows()-offset-block));
        Mid = multiply_self_transpose(L);
        Mid = subtract(Mid_temp, Mid);
        A.sub_block(Mid, 0, 0, (offset+block), (offset+block),
          (A.rows()-offset-block), (A.rows()-offset-block));
        offset += block;
      }
      int left = A.rows() - offset;
      if (left > 0) {
        matrix_gpu D(left, left);
        matrix_gpu V(left, left);
        D.sub_block(A, offset, offset, 0, 0, left, left);
        V.zeros();
        try {
          kernel_chol_block.setArg(0, V.buffer());
          kernel_chol_block.setArg(1, D.buffer());
          kernel_chol_block.setArg(2, left);
          cmd_queue.enqueueNDRangeKernel(kernel_chol_block,
           cl::NullRange, cl::NDRange(left), cl::NDRange(left));
        } catch (const cl::Error& e) {
          check_opencl_error("cholesky_decompose", e);
        }
        A.sub_block(V, 0, 0, offset, offset, left, left);
      }
      V.zeros<gpu::Upper>();
      A.triangular_transpose<gpu::LowerToUpper>();
      check_nan("cholesky_decompose_gpu",
        "Matrix m", A);
      check_diagonal_zeros("cholesky_decompose_gpu",
        "Matrix m", A);
      A.zeros<gpu::Upper>();
      matrix_gpu B(A);
      return B;
    }
     /**
     * Return the lower-triangular Cholesky factor (i.e., matrix
     * square root) of the specified square, symmetric matrix.  
     * The return value \f$L\f$ will be a lower-traingular matrix such that the
     * original matrix \f$A\f$ is given by
     * <p>\f$A = L \times L^T\f$.
     * The Cholesky decomposition is computed on the GPU. The
     * input matrix is transfered to the GPU and the resulting
     * lower-triangular matrix is then copied from the GPU.
     * 
     * @param m Symmetrix matrix.
     * @return Square root of matrix.
     * @throw std::domain_error if m is not a symmetric matrix or
     *   if m is not positive definite (if m has more than 0 elements)
     */
    template <typename T>
    typename boost::enable_if_c<boost::is_arithmetic<T>::value,
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>::type
    cholesky_decompose_gpu(const Eigen::Matrix<T,
     Eigen::Dynamic, Eigen::Dynamic>& m) {
      if (m.size() == 0) return m;
      matrix_gpu A(m);
      check_symmetric("cholesky_decompose", "m", A);

      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
       m_tmp(m.rows(), m.cols());
      int max_workgroup_size = opencl_context.max_workgroup_size();
      if (m.cols() <= 512) {
        A = cholesky_decompose_gpu(A, std::min(100, max_workgroup_size));
        copy(m_tmp, A); // NOLINT
        return m_tmp;
      }
      cl::CommandQueue cmd_queue = opencl_context.queue();
      cl::Kernel kernel_chol_block
          = opencl_context.get_kernel("cholesky_block");
      // Will be managed by the library core system
      int block = std::min(420, max_workgroup_size);
      int offset = 0;
      matrix_gpu V(block, block);
      matrix_gpu D(block, block);

      while ((offset + block) < (A.rows())) {
        matrix_gpu L(A.rows()-offset-block, block);
        matrix_gpu Mid(A.rows()-offset-block, A.rows()-offset-block);
        matrix_gpu Mid_temp(A.rows()-offset-block, A.rows()-offset-block);

        D.sub_block(A, offset, offset, 0, 0, block, block);
        V.zeros();
        int block_level2 = std::min(100, max_workgroup_size);
        V = cholesky_decompose_gpu(D, block_level2);
        copy(D, V);
        A.sub_block(V, 0, 0, offset, offset, block, block);

        V = lower_triangular_inverse(D);

        L.sub_block(A, (offset+block), offset, 0, 0,
          (A.rows()-offset-block) , block);
        V = transpose(V);
        L = multiply(L, V);
        A.sub_block(L, 0, 0, (offset+block), offset,
          (A.rows()-offset-block) , block);

        Mid_temp.sub_block(A, (offset+block), (offset+block),
          0, 0, (A.rows()-offset-block), (A.rows()-offset-block));
        Mid = multiply_self_transpose(L);
        Mid = subtract(Mid_temp, Mid);
        A.sub_block(Mid, 0, 0, (offset+block), (offset+block),
          (A.rows()-offset-block), (A.rows()-offset-block));

        offset += block;
      }
      int left = A.rows() - offset;
      if (left > 0) {
        matrix_gpu D(left, left);
        matrix_gpu V(left, left);
        D.sub_block(A, offset, offset, 0, 0, left, left);
        V.zeros();
        V = cholesky_decompose_gpu(D, 100);
        A.sub_block(V, 0, 0, offset, offset, left, left);
      }
      A.zeros<gpu::Upper>();
      A.triangular_transpose<gpu::LowerToUpper>();
      check_nan("cholesky_decompose_gpu",
        "Matrix m", A);
      check_diagonal_zeros("cholesky_decompose_gpu",
        "Matrix m", A);
      A.zeros<gpu::Upper>();
      copy(m_tmp, A); // NOLINT
      return m_tmp;
    }
  }
}

#endif
#endif
