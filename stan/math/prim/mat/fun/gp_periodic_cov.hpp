#ifndef STAN_MATH_PRIM_MAT_FUN_GP_PERIODIC_COV_HPP
#define STAN_MATH_PRIM_MAT_FUN_GP_PERIODIC_COV_HPP

#include <math.h>
#include <stan/math/prim/mat/fun/distance.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/distance.hpp>
#include <stan/math/prim/scal/fun/inv.hpp>
#include <stan/math/prim/scal/fun/inv_square.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <cmath>

#include <stan/math/gpu/matrix_gpu.hpp>
#include <stan/math/gpu/cov_exp_quad.hpp>
#include <vector>

namespace stan {
namespace math {
int x_on_gpu = 0;
matrix_gpu *x_gpu;
/**
 * Returns a periodic covariance matrix \f$ \mathbf{K} \f$ using the input \f$
 * \mathbf{X} \f$. The elements of \f$ \mathbf{K} \f$ are defined as \f$
 * \mathbf{K}_{ij} = k(\mathbf{X}_i,\mathbf{X}_j), \f$ where \f$ \mathbf{X}_i
 * \f$ is the \f$i\f$-th row of \f$ \mathbf{X} \f$ and \n \f$
 * k(\mathbf{x},\mathbf{x}^\prime) = \sigma^2 \exp\left(-\frac{2\sin^2(\pi
 * |\mathbf{x}-\mathbf{x}^\prime|/p)}{\ell^2}\right), \f$ \n where \f$ \sigma^2
 * \f$, \f$ \ell \f$ and \f$ p \f$ are the signal variance, length-scale and
 * period.
 *
 * @tparam T_x type of std::vector elements of x.
 *   T_x can be a scalar, an Eigen::Vector, or an Eigen::RowVector.
 * @tparam T_sigma type of sigma
 * @tparam T_l type of length-scale
 * @tparam T_p type of period
 *
 * @param x std::vector of input elements.
 *   This function assumes that all elements of x have the same size.
 * @param sigma standard deviation of the signal
 * @param l length-scale
 * @param p period
 * @return periodic covariance matrix
 * @throw std::domain_error if sigma <= 0, l <= 0, p <= 0 or
 *   x is nan or infinite
 */
template <typename T_x, typename T_sigma, typename T_l, typename T_p>
inline typename Eigen::Matrix<
    typename stan::return_type<T_x, T_sigma, T_l, T_p>::type, Eigen::Dynamic,
    Eigen::Dynamic>
gp_periodic_cov(const std::vector<T_x> &x, const T_sigma &sigma, const T_l &l,
                const T_p &p) {    

  using std::exp;
  const char *fun = "gp_periodic_cov";
  check_positive(fun, "signal standard deviation", sigma);
  check_positive(fun, "length-scale", l);
  check_positive(fun, "period", p);
  for (size_t n = 0; n < x.size(); ++n)
    check_not_nan(fun, "element of x", x[n]);

  Eigen::Matrix<typename stan::return_type<T_x, T_sigma, T_l, T_p>::type,
                Eigen::Dynamic, Eigen::Dynamic>
      cov(x.size(), x.size());
  
  Eigen::Matrix<double,
                Eigen::Dynamic, Eigen::Dynamic>
      cov2(x.size(), x.size());
  
  size_t x_size = x.size();
  if (x_size == 0)
    return cov;
 
  clock_t start = clock();
  T_sigma sigma_sq = square(sigma);
  T_l neg_two_inv_l_sq = -2.0 * inv_square(l);
  T_p pi_div_p = pi() / p;
  
#ifdef STAN_OPENCL
  if(x_on_gpu == 0){
      x_gpu = new matrix_gpu(x);
      x_on_gpu = 1;
  }
  matrix_gpu cov_gpu(x.size(), x.size());
  Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic>
      consts(3, 1);
  consts(0) = value_of(sigma_sq);
  consts(1) = value_of(neg_two_inv_l_sq);
  consts(2) = value_of(pi_div_p);
  matrix_gpu consts_gpu(consts);
  gp_periodic_cov(*x_gpu, consts_gpu, cov_gpu);
  
  copy(cov2, cov_gpu);  
  cov=cov2;
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"gp_periodic_cov a " <<  x.size() << "- " << duration*1000.0 << std::endl;;
  
#else
  for (size_t j = 0; j < x_size; ++j) {
    cov(j, j) = sigma_sq;
    for (size_t i = j + 1; i < x_size; ++i) {
      cov(i, j) = sigma_sq
                  * exp(square(sin(pi_div_p * distance(x[i], x[j])))
                        * neg_two_inv_l_sq);
      cov(j, i) = cov(i, j);
    }
  }
#endif
  
  return cov;
}

/**
 * Returns a periodic covariance matrix \f$ \mathbf{K} \f$ using inputs
 * \f$ \mathbf{X}_1 \f$ and \f$ \mathbf{X}_2 \f$.
 * The elements of \f$ \mathbf{K} \f$ are defined as
 * \f$ \mathbf{K}_{ij} = k(\mathbf{X}_{1_i},\mathbf{X}_{2_j}), \f$ where
 * \f$ \mathbf{X}_{1_i} \f$ and \f$ \mathbf{X}_{2_j} \f$  are the \f$i\f$-th and
 * \f$j\f$-th rows of \f$ \mathbf{X}_1 \f$ and \f$ \mathbf{X}_2 \f$ and \n \f$
 * k(\mathbf{x},\mathbf{x}^\prime) = \sigma^2 \exp\left(-\frac{2\sin^2(\pi
 * |\mathbf{x}-\mathbf{x}^\prime|/p)}{\ell^2}\right), \f$ \n where \f$ \sigma^2
 * \f$, \f$ \ell \f$ and \f$ p \f$ are the signal variance, length-scale and
 * period.
 *
 * @tparam T_x1 type of std::vector elements of x1
 *   T_x1 can be a scalar, an Eigen::Vector, or an Eigen::RowVector.
 * @tparam T_x2 type of std::vector elements of x2
 *   T_x2 can be a scalar, an Eigen::Vector, or an Eigen::RowVector.
 * @tparam T_sigma type of sigma
 * @tparam T_l type of length-scale
 * @tparam T_p type of period
 *
 * @param x1 std::vector of first input elements
 * @param x2 std::vector of second input elements.
 *   This function assumes that all the elements of x1 and x2 have the same
 * sizes.
 * @param sigma standard deviation of the signal
 * @param l length-scale
 * @param p period
 * @return periodic covariance matrix
 * @throw std::domain_error if sigma <= 0, l <= 0, p <= 0 ,
 *   x1 or x2 is nan or infinite
 */
template <typename T_x1, typename T_x2, typename T_sigma, typename T_l,
          typename T_p>
inline typename Eigen::Matrix<
    typename stan::return_type<T_x1, T_x2, T_sigma, T_l, T_p>::type,
    Eigen::Dynamic, Eigen::Dynamic>
gp_periodic_cov(const std::vector<T_x1> &x1, const std::vector<T_x2> &x2,
                const T_sigma &sigma, const T_l &l, const T_p &p) {
    clock_t start = clock();
  using std::exp;
  const char *fun = "gp_periodic_cov";
  check_positive(fun, "signal standard deviation", sigma);
  check_positive(fun, "length-scale", l);
  check_positive(fun, "period", p);
  for (size_t n = 0; n < x1.size(); ++n)
    check_not_nan(fun, "element of x1", x1[n]);
  for (size_t n = 0; n < x2.size(); ++n)
    check_not_nan(fun, "element of x2", x2[n]);

  Eigen::Matrix<typename stan::return_type<T_x1, T_x2, T_sigma, T_l, T_p>::type,
                Eigen::Dynamic, Eigen::Dynamic>
      cov(x1.size(), x2.size());
  if (x1.size() == 0 || x2.size() == 0)
    return cov;

  T_sigma sigma_sq = square(sigma);
  T_l neg_two_inv_l_sq = -2.0 * inv_square(l);
  T_p pi_div_p = pi() / p;

  for (size_t i = 0; i < x1.size(); ++i) {
    for (size_t j = 0; j < x2.size(); ++j) {
      cov(i, j) = sigma_sq
                  * exp(square(sin(pi_div_p * distance(x1[i], x2[j])))
                        * neg_two_inv_l_sq);
    }
  }
    clock_t stop = clock();
    double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<"gp_periodic_cov b " << duration*1000.0 << std::endl;;
  return cov;
}
}  // namespace math
}  // namespace stan

#endif
