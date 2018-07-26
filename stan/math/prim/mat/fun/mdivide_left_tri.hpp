#ifndef STAN_MATH_PRIM_MAT_FUN_MDIVIDE_LEFT_TRI_HPP
#define STAN_MATH_PRIM_MAT_FUN_MDIVIDE_LEFT_TRI_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/promote_common.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/gpu/matrix_gpu.hpp>
#include <stan/math/gpu/subtract.hpp>
#include <stan/math/gpu/multiply.hpp>
#include <stan/math/gpu/inverse_gpu.hpp>
#include <algorithm>

namespace stan {
namespace math {

/**
 * Returns the solution of the system Ax=b when A is triangular
 * @param A Triangular matrix.  Specify upper or lower with TriView
 * being Eigen::Upper or Eigen::Lower.
 * @param b Right hand side matrix or vector.
 * @return x = A^-1 b, solution of the linear system.
 * @throws std::domain_error if A is not square or the rows of b don't
 * match the size of A.
 */
template <int TriView, typename T1, typename T2, int R1, int C1, int R2, int C2>
inline Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type,
                     R1, C2>
mdivide_left_tri(const Eigen::Matrix<T1, R1, C1> &A,
                 const Eigen::Matrix<T2, R2, C2> &b) {
#ifdef STAN_OPENCL
  if ( TriView == Eigen::Lower ) {
    check_square("mdivide_left_tri", "A", A);
    check_multiplicable("mdivide_left_tri", "A", A, "b", b);
    matrix_gpu A_gpu(A);
    matrix_gpu b_gpu(b);
    A_gpu = lower_triangular_inverse(A_gpu);
    matrix_gpu c_gpu = multiply(A_gpu, b_gpu);
    int n = A.rows();
    Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type,
                    R1, C2> cc(c_gpu.rows(), c_gpu.cols());
    copy(cc, c_gpu);
    return cc;
  } else {
    check_square("mdivide_left_tri", "A", A);
    check_multiplicable("mdivide_left_tri", "A", A, "b", b);
    Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type,
      R1, C2> aa = promote_common<Eigen::Matrix<T1, R1, C1>,
          Eigen::Matrix<T2, R1, C1> >(A)
        .template triangularView<TriView>()
        .solve(
            promote_common<Eigen::Matrix<T1, R2, C2>,
             Eigen::Matrix<T2, R2, C2> >(b));
    return aa;
  }
#else
  check_square("mdivide_left_tri", "A", A);
  check_multiplicable("mdivide_left_tri", "A", A, "b", b);
  Eigen::Matrix<typename boost::math::tools::promote_args<T1,
    T2>::type, R1, C2> aa = promote_common<Eigen::Matrix<T1, R1, C1>,
     Eigen::Matrix<T2, R1, C1> >(A)
      .template triangularView<TriView>()
      .solve(
          promote_common<Eigen::Matrix<T1, R2, C2>,
           Eigen::Matrix<T2, R2, C2> >(b));
  return aa;
#endif
}

/**
 * Returns the solution of the system Ax=b when A is triangular and b=I.
 * @param A Triangular matrix.  Specify upper or lower with TriView
 * being Eigen::Upper or Eigen::Lower.
 * @return x = A^-1 .
 * @throws std::domain_error if A is not square
 */
template <int TriView, typename T, int R1, int C1>
inline Eigen::Matrix<T, R1, C1> mdivide_left_tri(
    const Eigen::Matrix<T, R1, C1> &A) {
#ifdef STAN_OPENCL
  matrix_gpu A_gpu(A);
  A_gpu = lower_triangular_inverse(A_gpu);
  int n = A.rows();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> c(n, n);
  copy(c, A_gpu);
  return c;
#else
  check_square("mdivide_left_tri", "A", A);
  int n = A.rows();
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b;
  b.setIdentity(n, n);
  A.template triangularView<TriView>().solveInPlace(b);
  return b;
#endif
}

}  // namespace math
}  // namespace stan
#endif
