#ifndef STAN_MATH_REV_MAT_FUN_MDIVIDE_LEFT_TRI_HPP
#define STAN_MATH_REV_MAT_FUN_MDIVIDE_LEFT_TRI_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>

namespace stan {
namespace math {

namespace internal {
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
#ifdef STAN_OPENCL
    if (A.rows()
        >= opencl_context.tuning_opts().tri_inverse_size_worth_transfer) {
      matrix_cl A_cl(A_, M_, M_);
      matrix_cl C_cl(C_, M_, N_);
      if (TriView == Eigen::Lower) {
        A_cl = tri_inverse<TriangularViewCL::Lower>(A_cl);
      } else {
        A_cl = tri_inverse<TriangularViewCL::Upper>(A_cl);
      }
      C_cl = A_cl * C_cl;
      C = from_matrix_cl(C_cl);
    } else {
#endif
      C = Map<Matrix<double, R1, C2> >(C_, M_, N_);
      C = Map<Matrix<double, R1, C1> >(A_, M_, M_)
                .template triangularView<TriView>()
                .solve(C);
#ifdef STAN_OPENCL
    }
#endif
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
#ifdef STAN_OPENCL
    if (M_
        >= opencl_context.tuning_opts().tri_inverse_size_worth_transfer) {
      matrix_cl A_cl(A_, M_, M_);
      matrix_cl C_cl(C_, M_, N_);
      matrix_cl adjC_cl(adjC);
      C_cl = transpose(C_cl);
      if (TriView == Eigen::Lower) {
        A_cl = transpose(tri_inverse<TriangularViewCL::Lower>(A_cl));
      } else {
        A_cl = transpose(tri_inverse<TriangularViewCL::Upper>(A_cl));
      }
      matrix_cl adjB_cl = A_cl * adjC_cl;
      matrix_cl adjA_cl = multiply(adjB_cl * C_cl, -1.0);
      adjA = from_matrix_cl(adjA_cl);
      adjB = from_matrix_cl(adjB_cl);
    } else {
#endif
    adjB = Map<Matrix<double, R1, C1> >(A_, M_, M_)
               .template triangularView<TriView>()
               .transpose()
               .solve(adjC);
    adjA.noalias()
        = -adjB * Map<Matrix<double, R1, C2> >(C_, M_, N_).transpose();
#ifdef STAN_OPENCL
    }
#endif
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
#ifdef STAN_OPENCL
    if (A.rows()
        >= opencl_context.tuning_opts().tri_inverse_size_worth_transfer) {
      matrix_cl A_cl(A_, M_, M_);
      matrix_cl C_cl(C_, M_, N_);
      if (TriView == Eigen::Lower) {
        A_cl = tri_inverse<TriangularViewCL::Lower>(A_cl);
      } else {
        A_cl = tri_inverse<TriangularViewCL::Upper>(A_cl);
      }
      C_cl = A_cl * C_cl;
      C = from_matrix_cl(C_cl);
    } else {
#endif
      C = Map<Matrix<double, R1, C2> >(C_, M_, N_);
      C = Map<Matrix<double, R1, C1> >(A_, M_, M_)
              .template triangularView<TriView>()
              .solve(C);
#ifdef STAN_OPENCL
    }
#endif
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
#ifdef STAN_OPENCL
    if (M_
        >= opencl_context.tuning_opts().tri_inverse_size_worth_transfer) {
      matrix_cl A_cl(A_, M_, M_);
      matrix_cl adjC_cl(adjC);
      if (TriView == Eigen::Lower) {
        A_cl = transpose(tri_inverse<TriangularViewCL::Lower>(A_cl));
      } else {
        A_cl = transpose(tri_inverse<TriangularViewCL::Upper>(A_cl));
      }
      matrix_cl adjB_cl = A_cl * adjC_cl;
      adjB = from_matrix_cl(adjB_cl);
    } else {
#endif
    adjB = Map<Matrix<double, R1, C1> >(A_, M_, M_)
               .template triangularView<TriView>()
               .transpose()
               .solve(adjC);
#ifdef STAN_OPENCL
    }
#endif
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
#ifdef STAN_OPENCL
    if (A.rows()
        >= opencl_context.tuning_opts().tri_inverse_size_worth_transfer) {
      matrix_cl A_cl(A_, M_, M_);
      matrix_cl B_cl(B);
      if (TriView == Eigen::Lower) {
        A_cl = tri_inverse<TriangularViewCL::Lower>(A_cl);
      } else {
        A_cl = tri_inverse<TriangularViewCL::Upper>(A_cl);
      }
      B_cl = A_cl * B_cl;
      C = from_matrix_cl(B_cl);
    } else {
#endif
    C = Map<Matrix<double, R1, C1> >(A_, M_, M_)
            .template triangularView<TriView>()
            .solve(B);
#ifdef STAN_OPENCL
    }
#endif
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
    Matrix<double, R1, C2> adjC(M_, N_);

    size_t pos = 0;
    for (size_type j = 0; j < adjC.cols(); j++)
      for (size_type i = 0; i < adjC.rows(); i++)
        adjC(i, j) = variRefC_[pos++]->adj_;
#ifdef STAN_OPENCL
    if (M_
        >= opencl_context.tuning_opts().tri_inverse_size_worth_transfer) {
      matrix_cl A_cl(A_, M_, M_);
      matrix_cl C_cl(C_, M_, N_);
      C_cl = transpose(C_cl);
      matrix_cl adjC_cl(adjC);
      if (TriView == Eigen::Lower) {
        A_cl = transpose(tri_inverse<TriangularViewCL::Lower>(A_cl));
      } else {
        A_cl = transpose(tri_inverse<TriangularViewCL::Upper>(A_cl));
      }
      matrix_cl adjA_cl = multiply(A_cl * (adjC_cl * C_cl), -1.0);
      adjA = from_matrix_cl(adjA_cl);
    } else {
#endif
    adjA.noalias()
        = -Map<Matrix<double, R1, C1> >(A_, M_, M_)
               .template triangularView<TriView>()
               .transpose()
               .solve(adjC
                      * Map<Matrix<double, R1, C2> >(C_, M_, N_).transpose());
#ifdef STAN_OPENCL
    }
#endif
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
}  // namespace internal

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
  internal::mdivide_left_tri_vv_vari<TriView, R1, C1, R2, C2> *baseVari
      = new internal::mdivide_left_tri_vv_vari<TriView, R1, C1, R2, C2>(A, b);

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
  internal::mdivide_left_tri_dv_vari<TriView, R1, C1, R2, C2> *baseVari
      = new internal::mdivide_left_tri_dv_vari<TriView, R1, C1, R2, C2>(A, b);

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
  internal::mdivide_left_tri_vd_vari<TriView, R1, C1, R2, C2> *baseVari
      = new internal::mdivide_left_tri_vd_vari<TriView, R1, C1, R2, C2>(A, b);

  size_t pos = 0;
  for (size_type j = 0; j < res.cols(); j++)
    for (size_type i = 0; i < res.rows(); i++)
      res(i, j).vi_ = baseVari->variRefC_[pos++];

  return res;
}

}  // namespace math
}  // namespace stan
#endif
