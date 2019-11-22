#ifndef STAN_MATH_PRIM_MAT_FUN_SYMMETRIC_EIGENSOLVER_HPP
#define STAN_MATH_PRIM_MAT_FUN_SYMMETRIC_EIGENSOLVER_HPP

#include <stan/math/prim/mat/fun/tridiagonalization.hpp>
#include <stan/math/prim/mat/fun/mrrr.hpp>

#include <Eigen/Dense>
#include <queue>

namespace stan {
namespace math {

/**
 * Calculates eigenvalues and eigenvectors of a selfadjoint matrix.
 * @param A The matrix
 * @param[out] eigenvalues sorted Eigenvalues.
 * @param[out] eigenvectors Eigenvectors - one per column.
 */
template <typename Scalar, typename Real = typename Eigen::NumTraits<Scalar>::Real>
void selfadjoint_eigensolver(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
                           Eigen::Matrix<Real, Eigen::Dynamic,1>& eigenvalues,
                           Eigen::Matrix<Scalar, Eigen::Dynamic,Eigen::Dynamic>& eigenvectors) {
  Eigen::Matrix<Scalar,Eigen::Dynamic, Eigen::Dynamic> packed = A;
  Eigen::Matrix<Scalar,Eigen::Dynamic, 1> hCoeffs(A.rows()-1);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tridiag_workspace(internal::block_householder_tridiag_workspace(A.rows()));
  internal::block_householder_tridiag_in_place(packed, hCoeffs, tridiag_workspace.data());
  Eigen::Matrix<Real,Eigen::Dynamic,1> diagonal = packed.diagonal().real();
  Eigen::Matrix<Scalar,Eigen::Dynamic,1> subdiagonal = packed.diagonal(-1).real();
  internal::tridiagonal_eigensolver(diagonal, subdiagonal, eigenvalues, eigenvectors);
  Eigen::HouseholderSequence<Eigen::Matrix<Scalar, Eigen::Dynamic,Eigen::Dynamic>, Eigen::Matrix<Scalar, Eigen::Dynamic,1>>(packed, hCoeffs.conjugate())
      .setLength(packed.rows() - 1)
      .setShift(1)
      .applyThisOnTheLeft(eigenvectors);
}

}  // namespace math
}  // namespace stan
#endif
