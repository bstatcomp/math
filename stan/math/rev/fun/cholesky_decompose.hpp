#ifndef STAN_MATH_REV_FUN_CHOLESKY_DECOMPOSE_HPP
#define STAN_MATH_REV_FUN_CHOLESKY_DECOMPOSE_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/value_of_rec.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/cholesky_decompose.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/to_ref.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of_rec.hpp>

#ifdef STAN_OPENCL
#include <stan/math/opencl/rev/opencl.hpp>
#endif

#include <algorithm>
#include <vector>

namespace stan {
namespace math {
namespace internal {

/**
 * Set the lower right triangular of a var matrix given a set of vari**
 *
 * @param L Matrix of vars
 * @param vari_ref Values to be set in lower right triangular of L.
 * @return None, L modified by reference.
 */
inline void set_lower_tri_coeff_ref(Eigen::Matrix<var, -1, -1>& L,
                                    vari** vari_ref) {
  size_t pos = 0;
  vari* dummy = new vari(0.0, false);

  for (size_type j = 0; j < L.cols(); ++j) {
    for (size_type i = j; i < L.cols(); ++i) {
      L.coeffRef(i, j).vi_ = vari_ref[pos++];
    }
    for (size_type k = 0; k < j; ++k) {
      L.coeffRef(k, j).vi_ = dummy;
    }
  }
  return;
}
}  // namespace internal

class cholesky_block : public vari {
 public:
  int M_;
  int block_size_;
  using Block_ = Eigen::Block<Eigen::MatrixXd>;
  vari** vari_ref_A_;
  vari** vari_ref_L_;

  /**
   * Constructor for Cholesky function.
   *
   * Stores varis for A.  Instantiates and stores varis for L.
   * Instantiates and stores dummy vari for upper triangular part of var
   * result returned in cholesky_decompose function call
   *
   * variRefL aren't on the chainable autodiff stack, only used for storage
   * and computation. Note that varis for L are constructed externally in
   * cholesky_decompose.
   *
   * block_size_ determined using the same calculation Eigen/LLT.h
   *
   * @param A matrix
   * @param L_A matrix, Cholesky factor of A
   */
  cholesky_block(const Eigen::Matrix<var, -1, -1>& A,
                 const Eigen::Matrix<double, -1, -1>& L_A)
      : vari(0.0),
        M_(A.rows()),
        vari_ref_A_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A.rows() * (A.rows() + 1) / 2)),
        vari_ref_L_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A.rows() * (A.rows() + 1) / 2)) {
    size_t pos = 0;
    block_size_ = std::max(M_ / 8, 8);
    block_size_ = std::min(block_size_, 128);
    for (size_type j = 0; j < M_; ++j) {
      for (size_type i = j; i < M_; ++i) {
        vari_ref_A_[pos] = A.coeffRef(i, j).vi_;
        vari_ref_L_[pos] = new vari(L_A.coeffRef(i, j), false);
        ++pos;
      }
    }
  }

  /**
   * Symbolic adjoint calculation for Cholesky factor A
   *
   * @param L Cholesky factor
   * @param L_adj matrix of adjoints of L
   */
  inline void symbolic_rev(Block_& L, Block_& L_adj) {
    using Eigen::Lower;
    using Eigen::StrictlyUpper;
    using Eigen::Upper;
    L.transposeInPlace();
    L_adj = (L * L_adj.triangularView<Lower>()).eval();
    L_adj.triangularView<StrictlyUpper>()
        = L_adj.adjoint().triangularView<StrictlyUpper>();
    L.triangularView<Upper>().solveInPlace(L_adj);
    L.triangularView<Upper>().solveInPlace(L_adj.transpose());
  }

  /**
   * Reverse mode differentiation algorithm reference:
   *
   * Iain Murray: Differentiation of the Cholesky decomposition, 2016.
   *
   */
  virtual void chain() {
    using Eigen::Block;
    using Eigen::Lower;
    using Eigen::MatrixXd;
    using Eigen::StrictlyUpper;
    using Eigen::Upper;
    auto L_adj = Eigen::MatrixXd::Zero(M_, M_).eval();
    auto L = Eigen::MatrixXd::Zero(M_, M_).eval();
    size_t pos = 0;
    for (size_type j = 0; j < M_; ++j) {
      for (size_type i = j; i < M_; ++i) {
        L_adj.coeffRef(i, j) = vari_ref_L_[pos]->adj_;
        L.coeffRef(i, j) = vari_ref_L_[pos]->val_;
        ++pos;
      }
    }

    for (int k = M_; k > 0; k -= block_size_) {
      int j = std::max(0, k - block_size_);
      Block_ R = L.block(j, 0, k - j, j);
      Block_ D = L.block(j, j, k - j, k - j);
      Block_ B = L.block(k, 0, M_ - k, j);
      Block_ C = L.block(k, j, M_ - k, k - j);
      Block_ R_adj = L_adj.block(j, 0, k - j, j);
      Block_ D_adj = L_adj.block(j, j, k - j, k - j);
      Block_ B_adj = L_adj.block(k, 0, M_ - k, j);
      Block_ C_adj = L_adj.block(k, j, M_ - k, k - j);
      if (C_adj.size() > 0) {
        C_adj = D.transpose()
                    .triangularView<Upper>()
                    .solve(C_adj.transpose())
                    .transpose();
        B_adj.noalias() -= C_adj * R;
        D_adj.noalias() -= C_adj.transpose() * C;
      }
      symbolic_rev(D, D_adj);
      R_adj.noalias() -= C_adj.transpose() * B;
      R_adj.noalias() -= D_adj.selfadjointView<Lower>() * R;
      D_adj.diagonal() *= 0.5;
      D_adj.triangularView<StrictlyUpper>().setZero();
    }
    pos = 0;
    for (size_type j = 0; j < M_; ++j) {
      for (size_type i = j; i < M_; ++i) {
        vari_ref_A_[pos++]->adj_ += L_adj.coeffRef(i, j);
      }
    }
  }
};

class cholesky_scalar : public vari {
 public:
  int M_;
  vari** vari_ref_A_;
  vari** vari_ref_L_;

  /**
   * Constructor for Cholesky function.
   *
   * Stores varis for A.  Instantiates and stores varis for L.  Instantiates
   * and stores dummy vari for upper triangular part of var result returned
   * in cholesky_decompose function call
   *
   * variRefL aren't on the chainable autodiff stack, only used for storage
   * and computation. Note that varis for L are constructed externally in
   * cholesky_decompose.
   *
   * @param A matrix
   * @param L_A matrix, Cholesky factor of A
   */
  cholesky_scalar(const Eigen::Matrix<var, -1, -1>& A,
                  const Eigen::Matrix<double, -1, -1>& L_A)
      : vari(0.0),
        M_(A.rows()),
        vari_ref_A_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A.rows() * (A.rows() + 1) / 2)),
        vari_ref_L_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A.rows() * (A.rows() + 1) / 2)) {
    size_t accum = 0;
    size_t accum_i = accum;
    for (size_type j = 0; j < M_; ++j) {
      for (size_type i = j; i < M_; ++i) {
        accum_i += i;
        size_t pos = j + accum_i;
        vari_ref_A_[pos] = A.coeffRef(i, j).vi_;
        vari_ref_L_[pos] = new vari(L_A.coeffRef(i, j), false);
      }
      accum += j;
      accum_i = accum;
    }
  }

  /**
   * Reverse mode differentiation algorithm reference:
   *
   * Mike Giles. An extended collection of matrix derivative results for
   * forward and reverse mode AD.  Jan. 2008.
   *
   * Note algorithm  as laid out in Giles is row-major, so Eigen::Matrices
   * are explicitly storage order RowMajor, whereas Eigen defaults to
   * ColumnMajor. Also note algorithm starts by calculating the adjoint for
   * A(M_ - 1, M_ - 1), hence pos on line 94 is decremented to start at pos
   * = M_ * (M_ + 1) / 2.
   */
  virtual void chain() {
    using Eigen::Matrix;
    using Eigen::RowMajor;
    Matrix<double, -1, -1, RowMajor> adjL(M_, M_);
    Matrix<double, -1, -1, RowMajor> LA(M_, M_);
    Matrix<double, -1, -1, RowMajor> adjA(M_, M_);
    size_t pos = 0;
    for (size_type i = 0; i < M_; ++i) {
      for (size_type j = 0; j <= i; ++j) {
        adjL.coeffRef(i, j) = vari_ref_L_[pos]->adj_;
        LA.coeffRef(i, j) = vari_ref_L_[pos]->val_;
        ++pos;
      }
    }

    --pos;
    for (int i = M_ - 1; i >= 0; --i) {
      for (int j = i; j >= 0; --j) {
        if (i == j) {
          adjA.coeffRef(i, j) = 0.5 * adjL.coeff(i, j) / LA.coeff(i, j);
        } else {
          adjA.coeffRef(i, j) = adjL.coeff(i, j) / LA.coeff(j, j);
          adjL.coeffRef(j, j)
              -= adjL.coeff(i, j) * LA.coeff(i, j) / LA.coeff(j, j);
        }
        for (int k = j - 1; k >= 0; --k) {
          adjL.coeffRef(i, k) -= adjA.coeff(i, j) * LA.coeff(j, k);
          adjL.coeffRef(j, k) -= adjA.coeff(i, j) * LA.coeff(i, k);
        }
        vari_ref_A_[pos--]->adj_ += adjA.coeffRef(i, j);
      }
    }
  }
};

#ifdef STAN_OPENCL
class cholesky_opencl : public vari {
 public:
  int M_;
  vari** vari_ref_A_;
  vari** vari_ref_L_;

  /**
   * Constructor for OpenCL Cholesky function.
   *
   * Stores varis for A.  Instantiates and stores varis for L.
   * Instantiates and stores dummy vari for upper triangular part of var
   * result returned in cholesky_decompose function call
   *
   * variRefL aren't on the chainable autodiff stack, only used for storage
   * and computation. Note that varis for L are constructed externally in
   * cholesky_decompose.
   *
   * @param A matrix
   * @param L_A Cholesky factor of A
   */
  cholesky_opencl(const Eigen::Matrix<var, -1, -1>& A,
                  const Eigen::Matrix<double, -1, -1>& L_A)
      : vari(0.0),
        M_(A.rows()),
        vari_ref_A_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A.rows() * (A.rows() + 1) / 2)),
        vari_ref_L_(ChainableStack::instance_->memalloc_.alloc_array<vari*>(
            A.rows() * (A.rows() + 1) / 2)) {
    size_t pos = 0;
    for (size_type j = 0; j < M_; ++j) {
      for (size_type i = j; i < M_; ++i) {
        vari_ref_A_[pos] = A.coeffRef(i, j).vi_;
        vari_ref_L_[pos] = new vari(L_A.coeffRef(i, j), false);
        ++pos;
      }
    }
  }

  /**
   * Symbolic adjoint calculation for Cholesky factor A
   *
   * @param L_val value of Cholesky factor
   * @param L_adj adjoint of Cholesky factor
   */
  inline void symbolic_rev(matrix_cl<double>& L_val, matrix_cl<double>& L_adj) {
    L_adj = transpose(L_val) * L_adj;
    L_adj.triangular_transpose<TriangularMapCL::LowerToUpper>();
    L_val = transpose(tri_inverse(L_val));
    L_adj = L_val * transpose(L_val * L_adj);
    L_adj.triangular_transpose<TriangularMapCL::LowerToUpper>();
  }

  /**
   * Reverse mode differentiation algorithm using OpenCL
   *
   * Reference:
   *
   * Iain Murray: Differentiation of the Cholesky decomposition, 2016.
   *
   */
  virtual void chain() {
    const int packed_size = M_ * (M_ + 1) / 2;
    Eigen::Map<Eigen::Matrix<vari*, Eigen::Dynamic, 1>> L_cpu(
        vari_ref_L_, M_ * (M_ + 1) / 2);
    Eigen::VectorXd L_val_cpu = L_cpu.val();
    Eigen::VectorXd L_adj_cpu = L_cpu.adj();
    matrix_cl<double> L_val = packed_copy<matrix_cl_view::Lower>(L_val_cpu, M_);
    matrix_cl<double> L_adj = packed_copy<matrix_cl_view::Lower>(L_adj_cpu, M_);
    int block_size
        = M_ / opencl_context.tuning_opts().cholesky_rev_block_partition;
    block_size = std::max(block_size, 8);
    block_size = std::min(
        block_size, opencl_context.tuning_opts().cholesky_rev_min_block_size);
    // The following is an OpenCL implementation of
    // the chain() function from the cholesky_block
    // vari class implementation
    for (int k = M_; k > 0; k -= block_size) {
      const int j = std::max(0, k - block_size);
      const int k_j_ind = k - j;
      const int m_k_ind = M_ - k;

      auto&& R_val = block(L_val, j, 0, k_j_ind, j);
      auto&& R_adj = block(L_adj, j, 0, k_j_ind, j);
      matrix_cl<double> D_val = block(L_val, j, j, k_j_ind, k_j_ind);
      matrix_cl<double> D_adj = block(L_adj, j, j, k_j_ind, k_j_ind);
      auto&& B_val = block(L_val, k, 0, m_k_ind, j);
      auto&& B_adj = block(L_adj, k, 0, m_k_ind, j);
      auto&& C_val = block(L_val, k, j, m_k_ind, k_j_ind);
      auto&& C_adj = block(L_adj, k, j, m_k_ind, k_j_ind);

      C_adj = C_adj * tri_inverse(D_val);
      B_adj = B_adj - C_adj * R_val;
      D_adj = D_adj - transpose(C_adj) * C_val;

      symbolic_rev(D_val, D_adj);

      R_adj = R_adj - transpose(C_adj) * B_val - D_adj * R_val;
      D_adj = diagonal_multiply(D_adj, 0.5);

      block(L_adj, j, j, k_j_ind, k_j_ind) = D_adj;
    }
    L_adj.view(matrix_cl_view::Lower);
    std::vector<double> L_adj_cpu_res = packed_copy(L_adj);
    for (size_type j = 0; j < packed_size; ++j) {
      vari_ref_A_[j]->adj_ += L_adj_cpu_res[j];
    }
  }
};
#endif

/**
 * Reverse mode specialization of Cholesky decomposition
 *
 * Internally calls Eigen::LLT rather than using
 * stan::math::cholesky_decompose in order to use an inplace decomposition.
 *
 * Note chainable stack varis are created below in Matrix<var, -1, -1>
 *
 * @param A Matrix
 * @return L Cholesky factor of A
 */
template <typename T, require_eigen_vt<is_var, T>* = nullptr>
inline Eigen::Matrix<var, T::RowsAtCompileTime, T::ColsAtCompileTime>
cholesky_decompose(const T& A) {
  const auto& A_ref = to_ref(A);
  Eigen::Matrix<double, T::RowsAtCompileTime, T::ColsAtCompileTime> L_A(
      value_of_rec(A_ref));
  check_not_nan("cholesky_decompose", "A", L_A);
#ifdef STAN_OPENCL
  L_A = cholesky_decompose(L_A);
#else
  check_symmetric("cholesky_decompose", "A", L_A);
  Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>, Eigen::Lower> L_factor(L_A);
  check_pos_definite("cholesky_decompose", "A", L_factor);
#endif
  // Memory allocated in arena.
  // cholesky_scalar gradient faster for small matrices compared to
  // cholesky_block
  vari* dummy = new vari(0.0, false);
  Eigen::Matrix<var, T::RowsAtCompileTime, T::ColsAtCompileTime> L(A.rows(),
                                                                   A.cols());
  if (L_A.rows() <= 35) {
    cholesky_scalar* baseVari = new cholesky_scalar(A_ref, L_A);
    size_t accum = 0;
    size_t accum_i = accum;
    for (size_type j = 0; j < L.cols(); ++j) {
      for (size_type i = j; i < L.cols(); ++i) {
        accum_i += i;
        size_t pos = j + accum_i;
        L.coeffRef(i, j).vi_ = baseVari->vari_ref_L_[pos];
      }
      for (size_type k = 0; k < j; ++k) {
        L.coeffRef(k, j).vi_ = dummy;
      }
      accum += j;
      accum_i = accum;
    }
  } else {
#ifdef STAN_OPENCL
    if (L_A.rows()
        > opencl_context.tuning_opts().cholesky_size_worth_transfer) {
      cholesky_opencl* baseVari = new cholesky_opencl(A_ref, L_A);
      internal::set_lower_tri_coeff_ref(L, baseVari->vari_ref_L_);
    } else {
      cholesky_block* baseVari = new cholesky_block(A_ref, L_A);
      internal::set_lower_tri_coeff_ref(L, baseVari->vari_ref_L_);
    }
#else
    cholesky_block* baseVari = new cholesky_block(A_ref, L_A);
    internal::set_lower_tri_coeff_ref(L, baseVari->vari_ref_L_);
#endif
  }

  return L;
}

}  // namespace math
}  // namespace stan
#endif
