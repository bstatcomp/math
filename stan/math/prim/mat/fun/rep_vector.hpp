#ifndef STAN_MATH_PRIM_MAT_FUN_REP_VECTOR_HPP
#define STAN_MATH_PRIM_MAT_FUN_REP_VECTOR_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

#ifdef STAN_OPENCL
#include <stan/math/opencl/rep_vector.hpp>
#include <stan/math/opencl/copy.hpp>
#endif
namespace stan {
namespace math {

template <typename T>
inline Eigen::Matrix<return_type_t<T>, Eigen::Dynamic, 1> rep_vector(const T& x,
                                                                     int n) {
  check_nonnegative("rep_vector", "n", n);
  return Eigen::Matrix<return_type_t<T>, Eigen::Dynamic, 1>::Constant(n, x);
}

inline Eigen::Matrix<double, Eigen::Dynamic, 1> rep_vector(const double& x,
                                                                     int n) {
  check_nonnegative("rep_vector", "n", n);
#ifdef STAN_OPENCL
  return from_matrix_cl(rep_vector_cl(x,n));
#else
  return Eigen::Matrix<return_type_t<T>, Eigen::Dynamic, 1>::Constant(n, x);
#endif
}

}  // namespace math
}  // namespace stan

#endif
