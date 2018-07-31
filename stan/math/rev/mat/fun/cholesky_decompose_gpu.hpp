#ifndef STAN_MATH_REV_MAT_FUN_CHOLESKY_DECOMPOSE_GPU_HPP
#define STAN_MATH_REV_MAT_FUN_CHOLESKY_DECOMPOSE_GPU_HPP
#ifdef STAN_OPENCL
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/gpu/matrix_gpu.hpp>
#include <stan/math/prim/mat/fun/cholesky_decompose_gpu.hpp>
#include <stan/math/gpu/diagonal_multiply.hpp>
#include <stan/math/gpu/multiply.hpp>
#include <stan/math/gpu/inverse_gpu.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
namespace stan {
  namespace math {

    class cholesky_gpu : public vari {
    public:
      int M_;
      vari** variRefA_;
      vari** variRefL_;


      /**
       * Constructor for GPU cholesky function.
       *
       * Stores varis for A.  Instantiates and stores varis for L.
       * Instantiates and stores dummy vari for upper triangular part of var
       * result returned in cholesky_decompose function call
       *
       * variRefL aren't on the chainable autodiff stack, only used for storage
       * and computation. Note that varis for L are constructed externally in
       * cholesky_decompose.
       *
       *
       * @param A matrix
       * @param L_A matrix, cholesky factor of A
       */
      cholesky_gpu(const Eigen::Matrix<var, -1, -1>& A,
                   const Eigen::Matrix<double, -1, -1>& L_A)
        : vari(0.0),
          M_(A.rows()),
          variRefA_(ChainableStack::instance().memalloc_.alloc_array<vari*>
                    (A.rows() * (A.rows() + 1) / 2)),
          variRefL_(ChainableStack::instance().memalloc_.alloc_array<vari*>
                    (A.rows() * (A.rows() + 1) / 2)) {
            size_t pos = 0;
            for (size_type j = 0; j < M_; ++j) {
              for (size_type i = j; i < M_; ++i) {
                variRefA_[pos] = A.coeffRef(i, j).vi_;
                variRefL_[pos] = new vari(L_A.coeffRef(i, j), false); ++pos;
              }
            }
          }

      /**
       * Reverse mode differentiation algorithm using a GPU
       * 
       * Reference:
       *
       * Iain Murray: Differentiation of the Cholesky decomposition, 2016.
       *
       */
      virtual void chain() {
        clock_t start = clock();
        using Eigen::MatrixXd;
        using Eigen::Lower;
        using Eigen::Block;
        using Eigen::Upper;
        using Eigen::StrictlyUpper;
        using Eigen::StrictlyLower;
        MatrixXd Lbar(M_, M_);
        MatrixXd L(M_, M_);
        Lbar.setZero();
        L.setZero();
        size_t pos = 0;
        for (size_type j = 0; j < M_; ++j) {
          for (size_type i = j; i < M_; ++i) {
            Lbar.coeffRef(i, j) = variRefL_[pos]->adj_;
            L.coeffRef(i, j) = variRefL_[pos]->val_;
            ++pos;
          }
        }
        matrix_gpu L_gpu(L);
        matrix_gpu Lbar_gpu(Lbar);
        int M = M_;
        int block_size_ = 128;
        block_size_ = std::max((M / 8 / 16) * 16, 8);
        block_size_ = std::min(block_size_, 512);
        if (M <= 256) {
          block_size_ = M;
        } else if (M <= 1024) {
          block_size_ = 256;
        } else if (M <= 4096) {
          block_size_ = 352;
        } else if (M <= 8192) {
          block_size_ = 352;
        }
        for (int k = M; k > 0; k -= block_size_) {
          int j = std::max(0, k - block_size_);
          matrix_gpu R_gpu(k-j, j);
          matrix_gpu D_gpu(k-j, k-j);
          matrix_gpu Dinv_gpu(k-j, k-j);
          matrix_gpu B_gpu(M-k, j);
          matrix_gpu C_gpu(M-k, k-j);

          matrix_gpu Rbar_gpu(k-j, j);
          matrix_gpu Dbar_gpu(k-j, k-j);
          matrix_gpu Dbar2_gpu(k-j, k-j);
          matrix_gpu Bbar_gpu(M-k, j);
          matrix_gpu Bbar2_gpu(M-k, j);
          matrix_gpu Cbar_gpu(M-k, k-j);
          matrix_gpu Cbar2_gpu(k-j, M-k);
          matrix_gpu Cbar3_gpu(k-j, M-k);
          matrix_gpu temp_gpu(k-j, j);

          R_gpu.sub_block(L_gpu, j, 0, 0, 0, k-j, j);
          D_gpu.sub_block(L_gpu, j, j, 0, 0, k-j, k-j);
          B_gpu.sub_block(L_gpu, k, 0, 0, 0, M-k, j);
          C_gpu.sub_block(L_gpu, k, j, 0, 0, M-k, k-j);

          Rbar_gpu.sub_block(Lbar_gpu, j, 0, 0, 0, k-j, j);
          Dbar_gpu.sub_block(Lbar_gpu, j, j, 0, 0, k-j, k-j);
          Bbar_gpu.sub_block(Lbar_gpu, k, 0, 0, 0, M-k, j);
          Cbar_gpu.sub_block(Lbar_gpu, k, j, 0, 0, M-k, k-j);

          if (Cbar_gpu.size() > 0) {
            copy(Dinv_gpu, D_gpu);
            Dinv_gpu = lower_triangular_inverse(Dinv_gpu);
            Dinv_gpu = transpose(Dinv_gpu);
            Cbar2_gpu = transpose(Cbar_gpu);

            Cbar3_gpu = multiply(Dinv_gpu, Cbar2_gpu);
            Cbar_gpu = transpose(Cbar3_gpu);

            Bbar2_gpu = multiply(Cbar_gpu, R_gpu);
            Bbar_gpu = subtract(Bbar_gpu, Bbar2_gpu);

            Cbar3_gpu = transpose(Cbar_gpu);
            Dbar2_gpu = multiply(Cbar3_gpu, C_gpu);
            Dbar_gpu = subtract(Dbar_gpu, Dbar2_gpu);
          }

          D_gpu = transpose(D_gpu);
          Dbar_gpu.zeros<gpu::Upper>();
          Dbar2_gpu = multiply(D_gpu, Dbar_gpu);
          Dbar2_gpu.triangular_transpose<gpu::LowerToUpper>();
          D_gpu = transpose(D_gpu);
          D_gpu = lower_triangular_inverse(D_gpu);
          D_gpu = transpose(D_gpu);
          Dbar_gpu = multiply(D_gpu, Dbar2_gpu);
          Dbar_gpu = transpose(Dbar_gpu);
          Dbar2_gpu = multiply(D_gpu, Dbar_gpu);

          if (Cbar_gpu.size() > 0 && B_gpu.size() > 0) {
            Cbar2_gpu = transpose(Cbar_gpu);
            temp_gpu = multiply(Cbar2_gpu, B_gpu);
            Rbar_gpu = subtract(Rbar_gpu, temp_gpu);
          }

          if (Dbar_gpu.size() > 0 && R_gpu.size() > 0) {
            copy(Dbar_gpu, Dbar2_gpu);
            Dbar_gpu.triangular_transpose<gpu::LowerToUpper>();
            temp_gpu = multiply(Dbar_gpu, R_gpu);
            Rbar_gpu = subtract(Rbar_gpu, temp_gpu);
          }
          Dbar2_gpu = diagonal_multiply(Dbar2_gpu, 0.5);
          Dbar2_gpu.zeros<gpu::Upper>();

          Lbar_gpu.sub_block(Rbar_gpu, 0, 0, j, 0, k-j, j);
          Lbar_gpu.sub_block(Dbar2_gpu, 0, 0, j, j, k-j, k-j);
          Lbar_gpu.sub_block(Bbar_gpu, 0, 0, k, 0, M-k, j);
          Lbar_gpu.sub_block(Cbar_gpu, 0, 0, k, j, M-k, k-j);
        }
        copy(Lbar, Lbar_gpu);
        pos = 0;
        for (size_type j = 0; j < M_; ++j)
          for (size_type i = j; i < M_; ++i)
            variRefA_[pos++]->adj_ += Lbar.coeffRef(i, j);
        clock_t stop = clock();
        double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
        std::cout<<"chol chain " << duration*1000.0 << std::endl;;
      }
    };
    /**
     * Reverse mode specialization of cholesky decomposition on a GPU
     *
     * Note chainable stack varis are created below in Matrix<var, -1, -1>
     *
     * @param A Matrix
     * @return L cholesky factor of A
     */
    inline Eigen::Matrix<var, -1, -1>
      cholesky_decompose_gpu(const Eigen::Matrix<var, -1, -1> &A) {
      clock_t start = clock();
      Eigen::Matrix<double, -1, -1> L_A(value_of_rec(A));
      L_A = cholesky_decompose_gpu(L_A);
      // Memory allocated in arena.
      vari* dummy = new vari(0.0, false);
      Eigen::Matrix<var, -1, -1> L(A.rows(), A.cols());
      cholesky_gpu *baseVari = new cholesky_gpu(A, L_A);
      size_t pos = 0;
      for (size_type j = 0; j < L.cols(); ++j) {
        for (size_type i = j; i < L.cols(); ++i) {
          L.coeffRef(i, j).vi_ = baseVari->variRefL_[pos++];
        }
        for (size_type k = 0; k < j; ++k)
          L.coeffRef(k, j).vi_ = dummy;
      }
      clock_t stop = clock();
      double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
      std::cout<<"cholesky base " << duration*1000.0 << std::endl;;
      return L;
    }
  }
}
#endif
#endif
