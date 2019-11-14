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
 * Tridiagonalize a selfadjoint matrix using block Housholder algorithm. A = Q *
 * T
 * * Q^T, where T is tridiagonal and Q is unitary.
 * @param A Input matrix
 * @param[out] packed Packed form of the tridiagonal matrix. Elements of the
 * resulting selfadjoint tridiagonal matrix T are in the diagonal and first
 * superdiagonal, which contains subdiagonal entries of T. Columns bellow
 * diagonal contain householder vectors that can be used to construct unitary
 * matrix Q.
 * @param r Block size. Affects only performance of the algorithm. Optimal value
 * depends on the size of A and cache of the processor. For larger matrices or
 * larger cache sizes a larger value is optimal.
 */
template <typename Scalar>
void block_householder_tridiag(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& packed,
    const Eigen::Index r = 60) {
  using Real = typename Eigen::NumTraits<Scalar>::Real;
  using MapType = typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::MapType;
  using MapTypeVec = typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::MapType;
  packed = A;
  Eigen::Index first_actual_r = std::min({r, static_cast<Eigen::Index>(packed.rows() - 2)});
  Eigen::Index V_size = packed.rows() * r;
  Eigen::Index partial_update_size = (packed.rows() - first_actual_r) * (packed.rows() - first_actual_r);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> workspace(V_size + partial_update_size);
  for (Eigen::Index k = 0; k < packed.rows() - 2; k += r) {
    const Eigen::Index actual_r = std::min({r, static_cast<Eigen::Index>(packed.rows() - k - 2)});
    MapType V(workspace.data(), packed.rows() - k, actual_r);

    for (Eigen::Index j = 0; j < actual_r; j++) {
      typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::ColXpr::SegmentReturnType householder = packed.col(k + j).tail(packed.rows() - k - j - 1);
      if (j != 0) {
        Eigen::Index householder_whole_size = packed.rows() - k - j;
        packed.col(k + j).tail(householder_whole_size) -= packed.block(k + j, k, householder_whole_size, j) * V.block(j, 0, 1, j).adjoint();
        packed.col(k + j).tail(householder_whole_size) -= V.block(j, 0, householder_whole_size, j) * packed.block(k + j, k, 1, j).adjoint();
      }
      Real q = householder.squaredNorm();
      Scalar alpha = -sqrt(q);
      if (householder[0] != Scalar(0.)) {
        alpha *= householder[0] / Eigen::numext::abs(householder[0]);
      }

      q -= Eigen::numext::abs2(householder[0]);
      householder[0] -= alpha;
      q += Eigen::numext::abs2(householder[0]);
      q = sqrt(q);
      if (q != 0.) {
        householder *= SQRT_2 / q;
      }

      typename MapType::ColXpr::SegmentReturnType v = V.col(j).tail(householder.size() + 1);
      v.tail(householder.size()).noalias() = packed.bottomRightCorner(packed.rows() - k - j - 1, packed.cols() - k - j - 1)
                    .template selfadjointView<Eigen::Lower>() * householder;
      MapTypeVec tmp(workspace.data() + V.size(), j);
      tmp.noalias() = V.bottomLeftCorner(householder.size(), j).adjoint() * householder;
      v.tail(householder.size()).noalias() -= packed.block(k + j + 1, k, householder.size(), j) * tmp;
      tmp.noalias() = packed.block(k + j + 1, k, householder.size(), j).adjoint() * householder;
      v.tail(householder.size()).noalias() -= V.bottomLeftCorner(householder.size(), j) * tmp;
      v[0] = q / SQRT_2;
      const Real cnst = (v.tail(householder.size()).adjoint() * householder).real()[0];
      v.tail(householder.size()) -= 0.5 * cnst * householder;

      // calculate subdiagonal of T into superdiagonal of packed
      packed(k + j, k + j + 1)
          = packed(k + j + 1, k + j) * q / SQRT_2 + alpha - v[0] * householder[0];
    }
    MapType partial_update(workspace.data() + V.size(), V.rows() - actual_r,V.rows() - actual_r);
    partial_update.noalias() = packed.block(k + actual_r, k, packed.rows() - k - actual_r, actual_r)
          * V.bottomRows(V.rows() - actual_r).adjoint();
    packed
        .block(k + actual_r, k + actual_r, packed.rows() - k - actual_r,
               packed.cols() - k - actual_r)
        .template triangularView<Eigen::Lower>()
        -= partial_update + partial_update.adjoint();
  }
  packed(packed.rows() - 2, packed.cols() - 1)
      = packed(packed.rows() - 1, packed.cols() - 2);
}

/**
 * Calculates Q*A in place. To construct Q pass identity matrix as input A.
 * @param packed Packed result of tridiagonalization that contains householder
 * vectors that define Q in columns bellow the diagonal. Usually result of a
 * call to `block_householder_tridiag`.
 * @param[in,out] A On input a matrix to multiply with Q. On output the product
 * Q*A.
 * @param r Block size. Affects only performance of the algorithm. Optimal value
 * depends on the size of A and cache of the processor. For larger matrices or
 * larger cache sizes a larger value is optimal.
 */
template <typename Scalar>
void block_apply_packed_Q(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& packed,
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& A,
    const Eigen::Index r = 100) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> workspace(A.rows(), r * 2);

  for (Eigen::Index k = (packed.rows() - 3) / r * r; k >= 0; k -= r) {
    const Eigen::Index actual_r = std::min({r, static_cast<Eigen::Index>(packed.rows() - k - 2)});
    typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::MapType W(workspace.data() + A.rows() * actual_r, packed.rows() - k - 1, actual_r);
    W.col(0) = packed.col(k).tail(W.rows());
    for (Eigen::Index j = 1; j < actual_r; j++) {
      workspace.col(0).head(j).noalias()
          = packed.block(k + j + 1, k, packed.rows() - k - j - 1, j).adjoint()
            * packed.col(k + j).tail(packed.rows() - k - j - 1);
      W.col(j).noalias() = -W.leftCols(j) * workspace.col(0).head(j);
      W.col(j).tail(W.rows() - j)
          += packed.col(k + j).tail(packed.rows() - k - j - 1);
    }
    workspace.transpose().topRows(actual_r).noalias()
        = packed.block(k + 1, k, packed.rows() - k - 1, actual_r)
              .adjoint()
              .template triangularView<Eigen::Upper>()
          * A.bottomRows(A.rows() - k - 1);
    A.bottomRows(A.cols() - k - 1).noalias()
        -= W * workspace.transpose().topRows(actual_r);
  }
}

}  // namespace internal
}  // namespace math
}  // namespace stan

#endif
