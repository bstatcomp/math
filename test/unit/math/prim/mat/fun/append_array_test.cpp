#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

TEST(MathFunctions, append_array_variadic_test) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> x(3), y(2), z(8), w(11), xx(12), yy(50), result;
  size_t expected_size = x.size() + y.size() + z.size() +  w.size() + xx.size() + yy.size();
  int i=0;
  for(int j=0;j<x.size();j++) {
    x[j] = i;
    i++;
  }
  for(int j=0;j<y.size();j++) {
    y[j] = i;
    i++;
  }
  for(int j=0;j<z.size();j++) {
    z[j] = i;
    i++;
  }
  for(int j=0;j<w.size();j++) {
    w[j] = i;
    i++;
  }
  for(int j=0;j<xx.size();j++) {
    xx[j] = i;
    i++;
  }
  for(int j=0;j<yy.size();j++) {
    yy[j] = i;
    i++;
  }
  EXPECT_NO_THROW(result = stan::math::append_array(x, y, z, w, xx, yy));
  EXPECT_EQ(expected_size, result.size());
}


TEST(MathFunctions, append_array_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> x(3), y(2), result;

  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = nan;
  y[0] = 0.5;
  y[1] = 1.0;

  EXPECT_NO_THROW(result = stan::math::append_array(x, y));
  EXPECT_PRED1(boost::math::isnan<double>, result[2]);

  EXPECT_NO_THROW(result = stan::math::append_array(y, x));
  EXPECT_PRED1(boost::math::isnan<double>, result[4]);
}

TEST(MathFunctions, append_array_check_size_vector1) {
  std::vector<std::vector<double> > x(3), y(3), result;

  for (size_t i = 0; i < x.size(); i++) {
    x[i].resize(3);
    y[i].resize(3);
  }
  EXPECT_NO_THROW(result = stan::math::append_array(x, y));

  for (size_t i = 0; i < x.size(); i++) {
    x[i].resize(3);
    y[i].resize(4);
  }
  EXPECT_THROW(result = stan::math::append_array(x, y), std::invalid_argument);
}

TEST(MathFunctions, append_array_check_size_vector2) {
  std::vector<std::vector<std::vector<double> > > x(3), y(3), result;

  for (size_t i = 0; i < 3; i++) {
    x[i].resize(3);
    y[i].resize(3);
    for (size_t j = 0; j < 3; j++) {
      x[i][j].resize(3);
      y[i][j].resize(3);
    }
  }
  EXPECT_NO_THROW(result = stan::math::append_array(x, y));

  for (size_t i = 0; i < 3; i++) {
    x[i].resize(3);
    y[i].resize(3);
    for (size_t j = 0; j < 3; j++) {
      x[i][j].resize(3);
      y[i][j].resize(4);
    }
  }
  EXPECT_THROW(result = stan::math::append_array(x, y), std::invalid_argument);

  for (size_t i = 0; i < 3; i++) {
    x[i].resize(4);
    y[i].resize(3);
    for (size_t j = 0; j < 3; j++) {
      x[i][j].resize(3);
      y[i][j].resize(3);
    }
  }
  EXPECT_THROW(result = stan::math::append_array(x, y), std::invalid_argument);
}
