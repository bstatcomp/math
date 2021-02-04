#ifndef STAN_MATH_OPENCL_PRIM_SUM_HPP
#define STAN_MATH_OPENCL_PRIM_SUM_HPP
#ifdef STAN_OPENCL

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/sum.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_generator.hpp>

namespace stan {
namespace math {

/**
 * Calculates sum of given kernel generator expression.
 * @tparam T type of the expression
 * @param m expression to sum
 * @return sum of given expression
 */
template <typename T,
          require_all_kernel_expressions_and_none_scalar_t<T>* = nullptr>
value_type_t<T> sum(const T& m) {
  return sum(from_matrix_cl(colwise_sum(reshape(m,m.size(),1))));
}

}  // namespace math
}  // namespace stan

#endif
#endif
