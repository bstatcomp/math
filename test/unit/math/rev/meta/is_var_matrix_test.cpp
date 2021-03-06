#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <gtest/gtest.h>

TEST(MetaTraitsRevScal, is_rev_matrix_test) {
  using stan::is_rev_matrix;
  using stan::math::var;
  using stan::math::var_value;
  using stan::math::vari;
  using stan::math::vari_value;
  EXPECT_TRUE((is_rev_matrix<Eigen::Matrix<var, -1, -1>>::value));
  EXPECT_TRUE((is_rev_matrix<Eigen::Matrix<var_value<float>, -1, -1>>::value));
  EXPECT_TRUE((is_rev_matrix<var_value<Eigen::MatrixXd>>::value));
  EXPECT_TRUE((is_rev_matrix<var_value<Eigen::SparseMatrix<double>>>::value));
  EXPECT_FALSE(is_rev_matrix<vari>::value);
  EXPECT_FALSE((is_rev_matrix<double>::value));
  EXPECT_FALSE((is_rev_matrix<vari_value<double>>::value));
  EXPECT_FALSE((is_rev_matrix<Eigen::MatrixXd>::value));
}
