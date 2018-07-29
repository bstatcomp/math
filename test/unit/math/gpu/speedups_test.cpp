#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/squared_distance.hpp>
#include <stan/math/prim/scal/fun/exp.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/math/distributions.hpp>
#include <vector>
TEST(SpeedupTests, matrix_multiply_dummy) {
  //ignore timing, just to compile the proper kernels
  int size = 100;
  Eigen::MatrixXd x(size, size);
  Eigen::MatrixXd y(size, size);
  Eigen::MatrixXd z(size, size);
  z = stan::math::multiply(x, y);  
}
TEST(MathMatrix, mdivide_left_tri_val) {
  using stan::math::mdivide_left_tri;
  int size = 50;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  I = mdivide_left_tri<Eigen::Lower>(Ad);
}

TEST(SpeedupTests, matrix_multiply_512) {
  int size = 512;
  Eigen::MatrixXd x(size, size);
  Eigen::MatrixXd y(size, size);
  Eigen::MatrixXd z(size, size);
  clock_t start = clock();
  z = stan::math::multiply(x, y);  
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"matrix multiply (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(SpeedupTests, matrix_multiply_1024) {
  int size = 1024;
  Eigen::MatrixXd x(size, size);
  Eigen::MatrixXd y(size, size);
  Eigen::MatrixXd z(size, size);
  clock_t start = clock();
  z = stan::math::multiply(x, y);  
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"matrix multiply (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(SpeedupTests, matrix_multiply_2048) {
  int size = 2048;
  Eigen::MatrixXd x(size, size);
  Eigen::MatrixXd y(size, size);
  Eigen::MatrixXd z(size, size);
  clock_t start = clock();
  z = stan::math::multiply(x, y);  
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"matrix multiply (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(SpeedupTests, matrix_multiply_4096) {
  int size = 4096;
  Eigen::MatrixXd x(size, size);
  Eigen::MatrixXd y(size, size);
  Eigen::MatrixXd z(size, size);
  clock_t start = clock();
  z = stan::math::multiply(x, y);  
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"matrix multiply (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(SpeedupTests, matrix_multiply_6144) {
  int size = 6144;
  Eigen::MatrixXd x(size, size);
  Eigen::MatrixXd y(size, size);
  Eigen::MatrixXd z(size, size);
  clock_t start = clock();
  z = stan::math::multiply(x, y);  
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"matrix multiply (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(SpeedupTests, matrix_multiply_8192) {
  int size = 8192;
  Eigen::MatrixXd x(size, size);
  Eigen::MatrixXd y(size, size);
  Eigen::MatrixXd z(size, size);
  clock_t start = clock();
  z = stan::math::multiply(x, y);  
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"matrix multiply (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, mdivide_left_tri_val10) {
  using stan::math::mdivide_left_tri;
  int size = 512;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  clock_t start = clock();
  I = mdivide_left_tri<Eigen::Lower>(Ad);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"mdivide_left_tri (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, mdivide_left_tri_val1) {
  using stan::math::mdivide_left_tri;
  int size = 1024;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  clock_t start = clock();
  I = mdivide_left_tri<Eigen::Lower>(Ad);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"mdivide_left_tri (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, mdivide_left_tri_val2) {
  using stan::math::mdivide_left_tri;
  int size = 2048;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  clock_t start = clock();
  I = mdivide_left_tri<Eigen::Lower>(Ad);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"mdivide_left_tri (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, mdivide_left_tri_val3) {
  using stan::math::mdivide_left_tri;
  int size = 4096;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  clock_t start = clock();
  I = mdivide_left_tri<Eigen::Lower>(Ad);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"mdivide_left_tri (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, mdivide_left_tri_val4) {
  using stan::math::mdivide_left_tri;
  int size = 6144;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  clock_t start = clock();
  I = mdivide_left_tri<Eigen::Lower>(Ad);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"mdivide_left_tri (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, mdivide_left_tri_val5) {
  using stan::math::mdivide_left_tri;
  int size = 8192;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  clock_t start = clock();
  I = mdivide_left_tri<Eigen::Lower>(Ad);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"mdivide_left_tri (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, mdivide_left_tri_val6) {
  using stan::math::mdivide_left_tri;
  int size = 10240;
  stan::math::matrix_d Ad(size, size);
  stan::math::matrix_d I;
  clock_t start = clock();
  I = mdivide_left_tri<Eigen::Lower>(Ad);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"mdivide_left_tri (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathPrimMat, vec_double_cov_exp_quad1) {
  int size = 512;
  double sigma = 0.2;
  double l = 5;
  
  std::vector<double> x(size);

  Eigen::MatrixXd cov;
  
  clock_t start = clock();
  cov = stan::math::cov_exp_quad(x, sigma, l);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"cov_exp_quad_x1 (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathPrimMat, vec_double_cov_exp_quad2) {
  int size = 1024;
  double sigma = 0.2;
  double l = 5;
  
  std::vector<double> x(size);

  Eigen::MatrixXd cov;
  
  clock_t start = clock();
  cov = stan::math::cov_exp_quad(x, sigma, l);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"cov_exp_quad_x1 (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathPrimMat, vec_double_cov_exp_quad3) {
  int size = 2048;
  double sigma = 0.2;
  double l = 5;
  
  std::vector<double> x(size);

  Eigen::MatrixXd cov;
  
  clock_t start = clock();
  cov = stan::math::cov_exp_quad(x, sigma, l);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"cov_exp_quad_x1 (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathPrimMat, vec_double_cov_exp_quad4) {
  int size = 4096;
  double sigma = 0.2;
  double l = 5;
  
  std::vector<double> x(size);

  Eigen::MatrixXd cov;
  
  clock_t start = clock();
  cov = stan::math::cov_exp_quad(x, sigma, l);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"cov_exp_quad_x1 (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathPrimMat, vec_double_cov_exp_quad6) {
  int size = 8192;
  double sigma = 0.2;
  double l = 5;
  
  std::vector<double> x(size);

  Eigen::MatrixXd cov;
  
  clock_t start = clock();
  cov = stan::math::cov_exp_quad(x, sigma, l);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"cov_exp_quad_x1 (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathPrimMat, vec_double_cov_exp_quad7) {
  int size = 8192*2;
  double sigma = 0.2;
  double l = 5;
  
  std::vector<double> x(size);

  Eigen::MatrixXd cov;
  
  clock_t start = clock();
  cov = stan::math::cov_exp_quad(x, sigma, l);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"cov_exp_quad_x1 (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathPrimMat, vec_double_cov_exp_quad8) {
  int size = 8192*4;
  double sigma = 0.2;
  double l = 5;
  
  std::vector<double> x(size);

  Eigen::MatrixXd cov;
  
  clock_t start = clock();
  cov = stan::math::cov_exp_quad(x, sigma, l);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"cov_exp_quad_x1 (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, cholesky_decompose_exception1) {
  int size = 512;
  stan::math::matrix_d m;

  m.resize(size, size);
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      m(i,j) = i%5;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 10.0;
  m=m*m.transpose();
  clock_t start = clock();
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, cholesky_decompose_exception2) {
  int size = 1024;
  stan::math::matrix_d m;

  m.resize(size, size);
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      m(i,j) = i%2;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 20.0;
  m=m*m.transpose();
  clock_t start = clock();
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, cholesky_decompose_exception3) {
  int size = 2048;
  stan::math::matrix_d m;

  m.resize(size, size);
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      m(i,j) = i%5;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 10.0;
  m=m*m.transpose();
  clock_t start = clock();
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, cholesky_decompose_exception4) {
  int size = 4096;
  stan::math::matrix_d m;

  m.resize(size, size);
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      m(i,j) = i%5;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 10.0;
  m=m*m.transpose();
  clock_t start = clock();
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, cholesky_decompose_exception5) {
  int size = 8192;
  stan::math::matrix_d m;

  m.resize(size, size);
  for(int i=0;i<size;i++){
    for(int j=0;j<size;j++){
      m(i,j) = i%5;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 10.0;
  m=m*m.transpose();
  clock_t start = clock();
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, cholesky_decompose_exception6) {
  int size = 512;
  stan::math::matrix_d m;
  boost::random::mt19937 rng;


  m.resize(size, size);
  /*for (int i = 0; i < size; i++)
  for (int j = 0; j < size; j++) {
    m(i, j) = stan::math::normal_rng(0.0, 1.0, rng);
  }
  m=m*m.transpose();*/
  
  clock_t start = clock();
  stan::math::cholesky_decompose_gpu(m);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose gpu (" << size << "):" << duration*1000.0 << std::endl;;
}

TEST(MathMatrix, cholesky_decompose_exception7) {
  int size = 1024;
  stan::math::matrix_d m;

  m.resize(size, size);
  /*for(int i=0;i<size;i++){
    for(int j=0;j<i;j++){
      m(i,j) = i%2;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 20.0;
  m=m*m.transpose();*/
  clock_t start = clock();
  stan::math::cholesky_decompose_gpu(m);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose gpu (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, cholesky_decompose_exception8) {
  int size = 2048;
  stan::math::matrix_d m;

  m.resize(size, size);
  /*for(int i=0;i<size;i++){
    for(int j=0;j<i;j++){
      m(i,j) = i%5;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 10.0;
  m=m*m.transpose();*/
  clock_t start = clock();
  stan::math::cholesky_decompose_gpu(m);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose gpu (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, cholesky_decompose_exception9) {
  int size = 4096;
  stan::math::matrix_d m;

  m.resize(size, size);
  /*for(int i=0;i<size;i++){
    for(int j=0;j<i;j++){
      m(i,j) = i%5;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 10.0;
  m=m*m.transpose();*/
  clock_t start = clock();
  stan::math::cholesky_decompose_gpu(m);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose gpu (" << size << "):" << duration*1000.0 << std::endl;;
}
TEST(MathMatrix, cholesky_decompose_exception10) {
  int size = 8192;
  stan::math::matrix_d m;

  m.resize(size, size);
  /*for(int i=0;i<size;i++){
    for(int j=0;j<i;j++){
      m(i,j) = i%5;
    }
  }
  for(int i=0;i<size;i++)
    m(i,i) = 10.0;
  m=m*m.transpose();*/
  
  clock_t start = clock();
  stan::math::cholesky_decompose_gpu(m,320, 440);
  clock_t stop = clock();
  double duration = ( stop - start ) / (double) CLOCKS_PER_SEC;
  std::cout<<"choleskky_decompose gpu ( " << size << " ):" << duration*1000.0 << std::endl;;

}
