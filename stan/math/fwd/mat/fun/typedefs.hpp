#ifndef STAN_MATH_FWD_MAT_FUN_TYPEDEFS_HPP
#define STAN_MATH_FWD_MAT_FUN_TYPEDEFS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/Eigen_NumTraits.hpp>

namespace stan {
namespace math {

using size_type = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Index;

using matrix_fd = Eigen::Matrix<fvar<double>, Eigen::Dynamic, Eigen::Dynamic>;

typedef Eigen::Matrix<fvar<fvar<double> >, Eigen::Dynamic, Eigen::Dynamic>
    matrix_ffd;

using vector_fd = Eigen::Matrix<fvar<double>, Eigen::Dynamic, 1>;

using vector_ffd = Eigen::Matrix<fvar<fvar<double> >, Eigen::Dynamic, 1>;

using row_vector_fd = Eigen::Matrix<fvar<double>, 1, Eigen::Dynamic>;

using row_vector_ffd = Eigen::Matrix<fvar<fvar<double> >, 1, Eigen::Dynamic>;

}  // namespace math
}  // namespace stan
#endif
