#ifndef STAN_MATH_PRIM_MAT_FUN_TRIDIAGONALIZATION_HPP
#define STAN_MATH_PRIM_MAT_FUN_TRIDIAGONALIZATION_HPP

#include <stan/math/prim/scal/fun/constants.hpp>
#include <Eigen/Dense>
#include <algorithm>
#include <ccomplex>
#include <cmath>

namespace stan {
namespace math {
namespace internal {

/**
 * Tridiagonalize a selfadjoint matrix using block Housholder algorithm.
 * A = Q * T * Q^T, where T is tridiagonal and Q is unitary.
 * @param[in,out] packed On input the input matrix. On output packed form of the
 * tridiagonal matrix. Elements of the resulting selfadjoint tridiagonal matrix
 * T are in the diagonal and first subdiagonal, which contains subdiagonal
 * entries of T. Householder coefficients are in the first superdiagonal.
 * Columns bellow diagonal contain householder vectors that can be used to
 * construct unitary matrix Q.
 * @param r Block size. Affects only performance of the algorithm. Optimal value
 * depends on the size of A and cache of the processor. For larger matrices or
 * larger cache sizes a larger value is optimal.
 */
template <typename Scalar>
void block_householder_tridiag_in_place(
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& packed,
    const Eigen::Index r = 60) {
  using Real = typename Eigen::NumTraits<Scalar>::Real;
  using MapType = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::MapType;
  using MapTypeVec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::MapType;
  Eigen::Index first_actual_r = std::min({r, static_cast<Eigen::Index>(packed.rows() - 1)});
  Eigen::Index V_size = (packed.rows() - 1) * r;
  Eigen::Index partial_update_size = (packed.rows() - first_actual_r) * (packed.rows() - first_actual_r);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> workspace(V_size + partial_update_size);
  for (Eigen::Index k = 0; k < packed.rows() - 1; k += r) {
    const Eigen::Index actual_r = std::min({r, static_cast<Eigen::Index>(packed.rows() - k - 1)});
    MapType V(workspace.data(), packed.rows() - k - 1, actual_r);

    for (Eigen::Index j = 0; j < actual_r; j++) {
      typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::ColXpr::SegmentReturnType householder = packed.col(k + j).tail(packed.rows() - k - j - 1);
      if (j != 0) {
        Eigen::Index householder_whole_size = packed.rows() - k - j;
        Scalar temp = packed(k + j, k + j - 1);
        packed(k + j, k + j - 1) = Scalar(1);
        packed.col(k + j).tail(householder_whole_size) -= packed.block(k + j, k, householder_whole_size, j) * V.block(j - 1, 0, 1, j).adjoint();
        packed.col(k + j).tail(householder_whole_size) -= V.block(j - 1, 0, householder_whole_size, j) * packed.block(k + j, k, 1, j).adjoint();
        packed(k + j, k + j - 1) = temp;
      }
      Scalar tau;
      Real beta;
      householder.makeHouseholderInPlace(tau, beta);
      householder[0] = Scalar(1);

      typename MapType::ColXpr::SegmentReturnType v = V.col(j).tail(householder.size());
      v.noalias() = packed.bottomRightCorner(packed.rows() - k - j - 1, packed.cols() - k - j - 1)
                    .template selfadjointView<Eigen::Lower>() * householder * Eigen::numext::conj(tau);
      MapTypeVec tmp(workspace.data() + V.size(), j);

      //Reminder of packed is not transformed by current block yet - v needs some fixes
      tmp.noalias() = V.bottomLeftCorner(householder.size(), j).adjoint() * householder * Eigen::numext::conj(tau);
      v.noalias() -= packed.block(k + j + 1, k, householder.size(), j) * tmp;
      tmp.noalias() = packed.block(k + j + 1, k, householder.size(), j).adjoint() * householder * Eigen::numext::conj(tau);
      v.noalias() -= V.bottomLeftCorner(householder.size(), j) * tmp;

      const Scalar cnst = (v.adjoint() * householder)[0];
      v -= Real(0.5)  * Eigen::numext::conj(tau) * cnst * householder;

      //store householder transformation scaling and subdiagonal of T
      packed(k + j, k + j + 1) = tau;
      packed(k + j + 1, k + j) = beta;
    }
    //update reminder of packed with the last block
    MapType partial_update(workspace.data() + V.size(), V.rows() - actual_r + 1,V.rows() - actual_r + 1);
    Scalar tmp = packed(k + actual_r, k+actual_r-1);
    packed(k + actual_r, k+actual_r-1) = Scalar(1);
    partial_update.noalias() = packed.block(k + actual_r, k, packed.rows() - k - actual_r, actual_r) * V.bottomRows(V.rows() - actual_r + 1).adjoint();
    packed(k + actual_r, k+actual_r-1) = tmp;
    packed.block(k + actual_r, k + actual_r, packed.rows() - k - actual_r, packed.cols() - k - actual_r).template triangularView<Eigen::Lower>()
        -= partial_update + partial_update.adjoint();
  }
}

}  // namespace internal
}  // namespace math
}  // namespace stan

#endif
