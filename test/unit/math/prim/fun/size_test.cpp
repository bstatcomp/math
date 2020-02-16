#include <stan/math/prim.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(MathPrimFun, size_scalar) {
  using stan::math::size;
  EXPECT_EQ(1U, size(27.0));
  EXPECT_EQ(1U, size(3));
}

TEST(MathPrimFun, size_vector) {
  using Eigen::Dynamic;
  using Eigen::Matrix;
  using stan::math::size;
  using std::vector;

  vector<int> x(5);
  EXPECT_EQ(5, size(x));

  vector<double> y(6);
  EXPECT_EQ(6, size(y));

  vector<Matrix<double, Dynamic, Dynamic> > z(7);
  EXPECT_EQ(7, size(z));

  vector<Matrix<double, Dynamic, 1> > a(8);
  EXPECT_EQ(8, size(a));

  vector<Matrix<double, 1, Dynamic> > b(9);
  EXPECT_EQ(9, size(b));

  vector<vector<double> > c(10);
  EXPECT_EQ(10, size(c));

  vector<vector<double> > ci(10);
  EXPECT_EQ(10, size(ci));

  vector<vector<Matrix<double, Dynamic, Dynamic> > > d(11);
  EXPECT_EQ(11, size(d));

  vector<vector<Matrix<double, 1, Dynamic> > > e(12);
  EXPECT_EQ(12, size(e));

  vector<vector<Matrix<double, Dynamic, 1> > > f(13);
  EXPECT_EQ(13, size(f));

  vector<vector<vector<Matrix<double, Dynamic, 1> > > > g(14);
  EXPECT_EQ(14, size(g));
}

TEST(MathPrimFun, size_matrices) {
  using stan::math::size;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m(2, 3);
  m << 1, 2, 3, 4, 5, 6;
  EXPECT_EQ(6U, size(m));

  Eigen::Matrix<double, Eigen::Dynamic, 1> rv(2);
  rv << 1, 2;
  EXPECT_EQ(2U, size(rv));

  Eigen::Matrix<double, 1, Eigen::Dynamic> v(2);
  v << 1, 2;
  EXPECT_EQ(2U, size(v));
}
