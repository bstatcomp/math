#ifndef STAN_MATH_PRIM_MAT_FUN_TYPEDEFS_HPP
#define STAN_MATH_PRIM_MAT_FUN_TYPEDEFS_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
namespace math {

/**
 * Type for sizes and indexes in an Eigen matrix with double e
 */
typedef index_type<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >::type
    size_type;

/**
 * Type for matrix of double values.
 */
using matrix_d = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * Type for (column) vector of double values.
 */
using vector_d = Eigen::Matrix<double, Eigen::Dynamic, 1>;

/**
 * Type for (row) vector of double values.
 */
using row_vector_d = Eigen::Matrix<double, 1, Eigen::Dynamic>;

}  // namespace math
}  // namespace stan

#endif
