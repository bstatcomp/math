#ifndef STAN_MATH_REV_MAT_FUN_MDIVIDE_LEFT_TRI_HPP
#define STAN_MATH_REV_MAT_FUN_MDIVIDE_LEFT_TRI_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/gpu/copy_triangular.hpp>
#include <stan/math/gpu/subtract.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <vector>
#include <algorithm>

namespace stan {
namespace math {

namespace {
template <int TriView, int R1, int C1, int R2, int C2>
class mdivide_left_tri_vv_vari : public vari {
 public:
  int M_;  // A.rows() = A.cols() = B.rows()
  int N_;  // B.cols()
  double *A_;
  double *C_;
  vari **variRefA_;
  vari **variRefB_;
  vari **variRefC_;

  mdivide_left_tri_vv_vari(const Eigen::Matrix<var, R1, C1> &A,
                           const Eigen::Matrix<var, R2, C2> &B)
      : vari(0.0),
        M_(A.rows()),
        N_(B.cols()),
        A_(reinterpret_cast<double *>(
            ChainableStack::instance().memalloc_.alloc(sizeof(double) * A.rows()
                                                       * A.cols()))),
        C_(reinterpret_cast<double *>(
            ChainableStack::instance().memalloc_.alloc(sizeof(double) * B.rows()
                                                       * B.cols()))),
        variRefA_(reinterpret_cast<vari **>(
            ChainableStack::instance().memalloc_.alloc(sizeof(vari *) * A.rows()
                                                       * (A.rows() + 1) / 2))),
        variRefB_(reinterpret_cast<vari **>(
            ChainableStack::instance().memalloc_.alloc(sizeof(vari *) * B.rows()
                                                       * B.cols()))),
        variRefC_(reinterpret_cast<vari **>(
            ChainableStack::instance().memalloc_.alloc(sizeof(vari *) * B.rows()
                                                       * B.cols()))) {
    using Eigen::Map;
    using Eigen::Matrix;

    size_t pos = 0;
    if (TriView == Eigen::Lower) {
      for (size_type j = 0; j < M_; j++)
        for (size_type i = j; i < M_; i++)
          variRefA_[pos++] = A(i, j).vi_;
    } else if (TriView == Eigen::Upper) {
      for (size_type j = 0; j < M_; j++)
        for (size_type i = 0; i < j + 1; i++)
          variRefA_[pos++] = A(i, j).vi_;
    }

    pos = 0;
    for (size_type j = 0; j < M_; j++) {
      for (size_type i = 0; i < M_; i++) {
        A_[pos++] = A(i, j).val();
      }
    }

    pos = 0;
    for (size_type j = 0; j < N_; j++) {
      for (size_type i = 0; i < M_; i++) {
        variRefB_[pos] = B(i, j).vi_;
        C_[pos++] = B(i, j).val();
      }
    }

    Matrix<double, R1, C2> C(M_, N_);
    C = Map<Matrix<double, R1, C2> >(C_, M_, N_);

    C = Map<Matrix<double, R1, C1> >(A_, M_, M_)
            .template triangularView<TriView>()
            .solve(C);

    pos = 0;
    for (size_type j = 0; j < N_; j++) {
      for (size_type i = 0; i < M_; i++) {
        C_[pos] = C(i, j);
        variRefC_[pos] = new vari(C_[pos], false);
        pos++;
      }
    }
  }

  virtual void chain() {
    using Eigen::Map;
    using Eigen::Matrix;
    Matrix<double, R1, C1> adjA(M_, M_);
    Matrix<double, R2, C2> adjB(M_, N_);
    Matrix<double, R1, C2> adjC(M_, N_);
    size_t pos = 0;
    for (size_type j = 0; j < adjC.cols(); j++)
      for (size_type i = 0; i < adjC.rows(); i++)
        adjC(i, j) = variRefC_[pos++]->adj_;

    adjB = Map<Matrix<double, R1, C1> >(A_, M_, M_)
               .template triangularView<TriView>()
               .transpose()
               .solve(adjC);
    adjA.noalias()
        = -adjB * Map<Matrix<double, R1, C2> >(C_, M_, N_).transpose();

    pos = 0;
    if (TriView == Eigen::Lower) {
      for (size_type j = 0; j < adjA.cols(); j++)
        for (size_type i = j; i < adjA.rows(); i++)
          variRefA_[pos++]->adj_ += adjA(i, j);
    } else if (TriView == Eigen::Upper) {
      for (size_type j = 0; j < adjA.cols(); j++)
        for (size_type i = 0; i < j + 1; i++)
          variRefA_[pos++]->adj_ += adjA(i, j);
    }

    pos = 0;
    for (size_type j = 0; j < adjB.cols(); j++)
      for (size_type i = 0; i < adjB.rows(); i++)
        variRefB_[pos++]->adj_ += adjB(i, j);
  }
};

template <int TriView, int R1, int C1, int R2, int C2>
class mdivide_left_tri_dv_vari : public vari {
 public:
  int M_;  // A.rows() = A.cols() = B.rows()
  int N_;  // B.cols()
  double *A_;
  double *C_;
  vari **variRefB_;
  vari **variRefC_;

  mdivide_left_tri_dv_vari(const Eigen::Matrix<double, R1, C1> &A,
                           const Eigen::Matrix<var, R2, C2> &B)
      : vari(0.0),
        M_(A.rows()),
        N_(B.cols()),
        A_(reinterpret_cast<double *>(
            ChainableStack::instance().memalloc_.alloc(sizeof(double) * A.rows()
                                                       * A.cols()))),
        C_(reinterpret_cast<double *>(
            ChainableStack::instance().memalloc_.alloc(sizeof(double) * B.rows()
                                                       * B.cols()))),
        variRefB_(reinterpret_cast<vari **>(
            ChainableStack::instance().memalloc_.alloc(sizeof(vari *) * B.rows()
                                                       * B.cols()))),
        variRefC_(reinterpret_cast<vari **>(
            ChainableStack::instance().memalloc_.alloc(sizeof(vari *) * B.rows()
                                                       * B.cols()))) {
    using Eigen::Map;
    using Eigen::Matrix;

    size_t pos = 0;
    for (size_type j = 0; j < M_; j++) {
      for (size_type i = 0; i < M_; i++) {
        A_[pos++] = A(i, j);
      }
    }

    pos = 0;
    for (size_type j = 0; j < N_; j++) {
      for (size_type i = 0; i < M_; i++) {
        variRefB_[pos] = B(i, j).vi_;
        C_[pos++] = B(i, j).val();
      }
    }

    Matrix<double, R1, C2> C(M_, N_);
    C = Map<Matrix<double, R1, C2> >(C_, M_, N_);

    C = Map<Matrix<double, R1, C1> >(A_, M_, M_)
            .template triangularView<TriView>()
            .solve(C);

    pos = 0;
    for (size_type j = 0; j < N_; j++) {
      for (size_type i = 0; i < M_; i++) {
        C_[pos] = C(i, j);
        variRefC_[pos] = new vari(C_[pos], false);
        pos++;
      }
    }
  }

  virtual void chain() {
    using Eigen::Map;
    using Eigen::Matrix;
    Matrix<double, R2, C2> adjB(M_, N_);
    Matrix<double, R1, C2> adjC(M_, N_);

    size_t pos = 0;
    for (size_type j = 0; j < adjC.cols(); j++)
      for (size_type i = 0; i < adjC.rows(); i++)
        adjC(i, j) = variRefC_[pos++]->adj_;

    adjB = Map<Matrix<double, R1, C1> >(A_, M_, M_)
               .template triangularView<TriView>()
               .transpose()
               .solve(adjC);

    pos = 0;
    for (size_type j = 0; j < adjB.cols(); j++)
      for (size_type i = 0; i < adjB.rows(); i++)
        variRefB_[pos++]->adj_ += adjB(i, j);
  }
};

template <int TriView, int R1, int C1, int R2, int C2>
class mdivide_left_tri_vd_vari : public vari {
 public:
  int M_;  // A.rows() = A.cols() = B.rows()
  int N_;  // B.cols()
  double *A_;
  double *C_;
  vari **variRefA_;
  vari **variRefC_;

  mdivide_left_tri_vd_vari(const Eigen::Matrix<var, R1, C1> &A,
                           const Eigen::Matrix<double, R2, C2> &B)
      : vari(0.0),
        M_(A.rows()),
        N_(B.cols()),
        A_(reinterpret_cast<double *>(
            ChainableStack::instance().memalloc_.alloc(sizeof(double) * A.rows()
                                                       * A.cols()))),
        C_(reinterpret_cast<double *>(
            ChainableStack::instance().memalloc_.alloc(sizeof(double) * B.rows()
                                                       * B.cols()))),
        variRefA_(reinterpret_cast<vari **>(
            ChainableStack::instance().memalloc_.alloc(sizeof(vari *) * A.rows()
                                                       * (A.rows() + 1) / 2))),
        variRefC_(reinterpret_cast<vari **>(
            ChainableStack::instance().memalloc_.alloc(sizeof(vari *) * B.rows()
                                                       * B.cols()))) {
    using Eigen::Map;
    using Eigen::Matrix;

    size_t pos = 0;
    if (TriView == Eigen::Lower) {
      for (size_type j = 0; j < M_; j++)
        for (size_type i = j; i < M_; i++)
          variRefA_[pos++] = A(i, j).vi_;
    } else if (TriView == Eigen::Upper) {
      for (size_type j = 0; j < M_; j++)
        for (size_type i = 0; i < j + 1; i++)
          variRefA_[pos++] = A(i, j).vi_;
    }

    pos = 0;
    for (size_type j = 0; j < M_; j++) {
      for (size_type i = 0; i < M_; i++) {
        A_[pos++] = A(i, j).val();
      }
    }

    Matrix<double, R1, C2> C(M_, N_);
    C = Map<Matrix<double, R1, C1> >(A_, M_, M_)
            .template triangularView<TriView>()
            .solve(B);

    pos = 0;
    for (size_type j = 0; j < N_; j++) {
      for (size_type i = 0; i < M_; i++) {
        C_[pos] = C(i, j);
        variRefC_[pos] = new vari(C_[pos], false);
        pos++;
      }
    }
  }

  virtual void chain() {
    using Eigen::Matrix;
    using Eigen::MatrixXd;
    using Eigen::Map;
    Matrix<double, R1, C1> adjA(M_, M_);
    Matrix<double, R1, C1> adjA1(M_, M_);
    Matrix<double, R1, C2> adjC(M_, N_);

    size_t pos = 0;
    for (size_type j = 0; j < adjC.cols(); j++)
      for (size_type i = 0; i < adjC.rows(); i++)
        adjC(i, j) = variRefC_[pos++]->adj_;

    if (TriView == Eigen::Lower) {
#ifndef STAN_OPENCL
      adjA.noalias()
        = -Map<Matrix<double, R1, C1> >(A_, M_, M_)
               .template triangularView<TriView>()
               .transpose()
               .solve(adjC
                      * Map<Matrix<double, R1, C2> >(C_, M_, N_).transpose());

#else

      Matrix<double, R1, C2> temp(M_, N_);
      Matrix<double, R1, C1> temp1(M_, M_);
      temp = Map<Matrix<double, R1, C2> >(C_, M_, N_);
      temp1 = Map<Matrix<double, R1, C1> >(A_, M_, M_);

      stan::math::matrix_gpu A4_gpu(adjC);
      stan::math::matrix_gpu A2_gpu(temp);
      stan::math::matrix_gpu A2t_gpu(N_, M_);
      stan::math::matrix_gpu AA_gpu(M_, M_);
      stan::math::matrix_gpu A3_gpu(M_, M_);
      stan::math::matrix_gpu A3a_gpu(M_, M_);

      A2t_gpu = stan::math::transpose(A2_gpu);
      AA_gpu = stan::math::multiply(A4_gpu, A2t_gpu);
      stan::math::matrix_gpu A1_gpu(temp1);
      A1_gpu = stan::math::copy_triangular<stan::math::gpu::Lower>(A1_gpu);
      A1_gpu = stan::math::lower_triangular_inverse(A1_gpu);
      A1_gpu = stan::math::transpose(A1_gpu);
      A3a_gpu = stan::math::multiply(A1_gpu, AA_gpu);
      A3a_gpu = stan::math::multiply(A3a_gpu, -1.0);
      stan::math::copy(adjA, A3a_gpu);
#endif
    } else {
      adjA.noalias()
        = -Map<Matrix<double, R1, C1> >(A_, M_, M_)
               .template triangularView<TriView>()
               .transpose()
               .solve(adjC
                      * Map<Matrix<double, R1, C2> >(C_, M_, N_).transpose());
    }
    pos = 0;
    if (TriView == Eigen::Lower) {
      for (size_type j = 0; j < adjA.cols(); j++)
        for (size_type i = j; i < adjA.rows(); i++)
          variRefA_[pos++]->adj_ += adjA(i, j);
    } else if (TriView == Eigen::Upper) {
      for (size_type j = 0; j < adjA.cols(); j++)
        for (size_type i = 0; i < j + 1; i++)
          variRefA_[pos++]->adj_ += adjA(i, j);
    }
  }
};
}  // namespace

template <int TriView, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<var, R1, C2> mdivide_left_tri(
    const Eigen::Matrix<var, R1, C1> &A, const Eigen::Matrix<var, R2, C2> &b) {
  Eigen::Matrix<var, R1, C2> res(b.rows(), b.cols());

  check_square("mdivide_left_tri", "A", A);
  check_multiplicable("mdivide_left_tri", "A", A, "b", b);

  // NOTE: this is not a memory leak, this vari is used in the
  // expression graph to evaluate the adjoint, but is not needed
  // for the returned matrix.  Memory will be cleaned up with the
  // arena allocator.
  mdivide_left_tri_vv_vari<TriView, R1, C1, R2, C2> *baseVari
      = new mdivide_left_tri_vv_vari<TriView, R1, C1, R2, C2>(A, b);

  size_t pos = 0;
  for (size_type j = 0; j < res.cols(); j++)
    for (size_type i = 0; i < res.rows(); i++)
      res(i, j).vi_ = baseVari->variRefC_[pos++];

  return res;
}
template <int TriView, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<var, R1, C2> mdivide_left_tri(
    const Eigen::Matrix<double, R1, C1> &A,
    const Eigen::Matrix<var, R2, C2> &b) {
  Eigen::Matrix<var, R1, C2> res(b.rows(), b.cols());

  check_square("mdivide_left_tri", "A", A);
  check_multiplicable("mdivide_left_tri", "A", A, "b", b);

  // NOTE: this is not a memory leak, this vari is used in the
  // expression graph to evaluate the adjoint, but is not needed
  // for the returned matrix.  Memory will be cleaned up with the
  // arena allocator.
  mdivide_left_tri_dv_vari<TriView, R1, C1, R2, C2> *baseVari
      = new mdivide_left_tri_dv_vari<TriView, R1, C1, R2, C2>(A, b);

  size_t pos = 0;
  for (size_type j = 0; j < res.cols(); j++)
    for (size_type i = 0; i < res.rows(); i++)
      res(i, j).vi_ = baseVari->variRefC_[pos++];

  return res;
}
template <int TriView, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<var, R1, C2> mdivide_left_tri(
    const Eigen::Matrix<var, R1, C1> &A,
    const Eigen::Matrix<double, R2, C2> &b) {
  Eigen::Matrix<var, R1, C2> res(b.rows(), b.cols());

  check_square("mdivide_left_tri", "A", A);
  check_multiplicable("mdivide_left_tri", "A", A, "b", b);

  // NOTE: this is not a memory leak, this vari is used in the
  // expression graph to evaluate the adjoint, but is not needed
  // for the returned matrix.  Memory will be cleaned up with the
  // arena allocator.
  mdivide_left_tri_vd_vari<TriView, R1, C1, R2, C2> *baseVari
      = new mdivide_left_tri_vd_vari<TriView, R1, C1, R2, C2>(A, b);

  size_t pos = 0;
  for (size_type j = 0; j < res.cols(); j++)
    for (size_type i = 0; i < res.rows(); i++)
      res(i, j).vi_ = baseVari->variRefC_[pos++];

  return res;
}

}  // namespace math
}  // namespace stan
#endif
