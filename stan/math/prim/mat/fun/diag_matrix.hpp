#ifndef STAN_MATH_PRIM_MAT_FUN_DIAG_MATRIX_HPP
#define STAN_MATH_PRIM_MAT_FUN_DIAG_MATRIX_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

#ifdef STAN_OPENCL
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/diag_matrix.hpp>
#include <stan/math/opencl/copy.hpp>
#endif
namespace stan {
namespace math {

/**
 * Return a square diagonal matrix with the specified vector of
 * coefficients as the diagonal values.
 * @param[in] v Specified vector.
 * @return Diagonal matrix with vector as diagonal values.
 */
template <typename T>
inline Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> diag_matrix(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& v) {
  return v.asDiagonal();
}

inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> diag_matrix(
        const Eigen::Matrix<double, Eigen::Dynamic, 1>& v) {
#ifdef STAN_OPENCL
  matrix_cl<double> v_cl(v);
  return from_matrix_cl(diag_matrix(v_cl));
#else
  return v.asDiagonal();
#endif
}

}  // namespace math
}  // namespace stan
#endif
