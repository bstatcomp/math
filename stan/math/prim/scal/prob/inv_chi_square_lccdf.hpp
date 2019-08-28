#ifndef STAN_MATH_PRIM_SCAL_PROB_INV_CHI_SQUARE_LCCDF_HPP
#define STAN_MATH_PRIM_SCAL_PROB_INV_CHI_SQUARE_LCCDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/fun/size_zero.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/gamma_p.hpp>
#include <stan/math/prim/scal/fun/digamma.hpp>
#include <stan/math/prim/scal/fun/tgamma.hpp>
#include <stan/math/prim/scal/fun/grad_reg_inc_gamma.hpp>
#include <cmath>
#include <limits>

namespace stan {
namespace math {

/**
 * Returns the inverse chi square log complementary cumulative distribution
 * function for the given variate and degrees of freedom. If given
 * containers of matching sizes, returns the log sum of probabilities.
 *
 * @tparam T_y type of scalar parameter
 * @tparam T_dof type of degrees of freedom parameter
 * @param y scalar parameter
 * @param nu degrees of freedom parameter
 * @return log probability or log sum of probabilities
 * @throw std::domain_error if y is negative or nu is nonpositive
 * @throw std::invalid_argument if container sizes mismatch
 */
template <typename T_y, typename T_dof>
return_type_t<T_y, T_dof> inv_chi_square_lccdf(const T_y& y, const T_dof& nu) {
  typedef partials_return_type_t<T_y, T_dof> T_partials_return;

  if (size_zero(y, nu)) {
    return 0.0;
  }

  static const char* function = "inv_chi_square_lccdf";

  T_partials_return P(0.0);

  check_positive_finite(function, "Degrees of freedom parameter", nu);
  check_not_nan(function, "Random variable", y);
  check_nonnegative(function, "Random variable", y);
  check_consistent_sizes(function, "Random variable", y,
                         "Degrees of freedom parameter", nu);

  scalar_seq_view<T_y> y_vec(y);
  scalar_seq_view<T_dof> nu_vec(nu);
  size_t N = max_size(y, nu);

  operands_and_partials<T_y, T_dof> ops_partials(y, nu);

  // Explicit return for extreme values
  // The gradients are technically ill-defined, but treated as zero
  for (size_t i = 0; i < stan::length(y); i++) {
    if (value_of(y_vec[i]) == 0) {
      return ops_partials.build(0.0);
    }
  }

  using std::exp;
  using std::log;
  using std::pow;

  VectorBuilder<!is_constant_all<T_dof>::value, T_partials_return, T_dof>
      gamma_vec(stan::length(nu));
  VectorBuilder<!is_constant_all<T_dof>::value, T_partials_return, T_dof>
      digamma_vec(stan::length(nu));

  if (!is_constant_all<T_dof>::value) {
    for (size_t i = 0; i < stan::length(nu); i++) {
      const T_partials_return nu_dbl = value_of(nu_vec[i]);
      gamma_vec[i] = tgamma(0.5 * nu_dbl);
      digamma_vec[i] = digamma(0.5 * nu_dbl);
    }
  }

  for (size_t n = 0; n < N; n++) {
    // Explicit results for extreme values
    // The gradients are technically ill-defined, but treated as zero
    if (value_of(y_vec[n]) == std::numeric_limits<double>::infinity()) {
      return ops_partials.build(negative_infinity());
    }

    const T_partials_return y_dbl = value_of(y_vec[n]);
    const T_partials_return y_inv_dbl = 1.0 / y_dbl;
    const T_partials_return nu_dbl = value_of(nu_vec[n]);

    const T_partials_return Pn = gamma_p(0.5 * nu_dbl, 0.5 * y_inv_dbl);

    P += log(Pn);

    if (!is_constant_all<T_y>::value) {
      ops_partials.edge1_.partials_[n]
          -= 0.5 * y_inv_dbl * y_inv_dbl * exp(-0.5 * y_inv_dbl)
             * pow(0.5 * y_inv_dbl, 0.5 * nu_dbl - 1) / tgamma(0.5 * nu_dbl)
             / Pn;
    }
    if (!is_constant_all<T_dof>::value) {
      ops_partials.edge2_.partials_[n]
          -= 0.5
             * grad_reg_inc_gamma(0.5 * nu_dbl, 0.5 * y_inv_dbl, gamma_vec[n],
                                  digamma_vec[n])
             / Pn;
    }
  }
  return ops_partials.build(P);
}

}  // namespace math
}  // namespace stan
#endif
