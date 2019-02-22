#include <stan/math/fwd/scal.hpp>
#include <gtest/gtest.h>

TEST(Fwd, distance) {
	using stan::math::fvar;
	using stan::math::distance;
	
	fvar<double> x1(1.0,1.0);
	fvar<double> x2(4.0,1.0);
	
	double x1c = 1.0;
	double x2c = 4.0;
	
	fvar<double> dx1 = distance(x1,x2c);
	EXPECT_FLOAT_EQ(3, dx1.val_);
    EXPECT_FLOAT_EQ(-1,dx1.d_);
	
	fvar<double> dx2 = distance(x1c,x2);
	EXPECT_FLOAT_EQ(3, dx2.val_);
    EXPECT_FLOAT_EQ(1,dx2.d_);
	
	dx1= distance(x1,x1c);
	EXPECT_FLOAT_EQ(0,dx1.val_);
	EXPECT_TRUE(stan::math::is_nan(dx1.d_));
	
	dx1= distance(x1c,x1);
	EXPECT_FLOAT_EQ(0,dx1.val_);
	EXPECT_TRUE(stan::math::is_nan(dx1.d_));
	
	
}

TEST(Fwd, distance_nan) {
  using stan::math::fvar;
  using stan::math::distance;
  double x1 = 1;
  double nan = std::numeric_limits<double>::quiet_NaN();

  fvar<double> x2(nan,1.0);
  
  EXPECT_THROW(distance(x1, x2), std::domain_error);
  EXPECT_THROW(distance(x2, x1), std::domain_error);
  EXPECT_THROW(distance(x2, x2), std::domain_error);

}

TEST(Fwd, distance_inf) {
  using stan::math::fvar;
  using stan::math::distance;
  double x1 = 1;
  double inf = std::numeric_limits<double>::infinity();

  fvar<double> x2(inf,1.0);
  
  EXPECT_THROW(distance(x1, x2), std::domain_error);
  EXPECT_THROW(distance(x2, x1), std::domain_error);
  EXPECT_THROW(distance(x2, x2), std::domain_error);
}
