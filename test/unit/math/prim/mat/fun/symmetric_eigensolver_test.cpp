#include <stan/math/prim/mat/fun/symmetric_eigensolver.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, selfadjoint_eigensolver_trivial_float) {
  Eigen::MatrixXf input(3, 3);
  input << 3, 0, 0, 0, 2, 0, 0, 0, 1;
  Eigen::VectorXf sortedDiag(3);
  sortedDiag << 1,2,3;

  Eigen::VectorXf eigenvals;
  Eigen::MatrixXf eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_TRUE(eigenvals.isApprox(sortedDiag));
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXf::Identity(3, 3)));
}

TEST(MathMatrix, selfadjoint_eigensolver_small_float) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXf input = Eigen::MatrixXf::Random(size, size);
  input += input.transpose().eval();

  Eigen::VectorXf eigenvals;
  Eigen::MatrixXf eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_NEAR(input.diagonal().sum(), eigenvals.sum(), 1e-6);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXf::Identity(size, size)));
  EXPECT_TRUE((input * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal()));
}

TEST(MathMatrix, selfadjoint_eigensolver_large_float) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXf input = Eigen::MatrixXf::Random(size, size);
  input += input.transpose().eval();

  Eigen::VectorXf eigenvals;
  Eigen::MatrixXf eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_NEAR(input.diagonal().sum(), eigenvals.sum(), 1e-3);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXf::Identity(size, size), 1e-2));
  EXPECT_TRUE((input * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal(), 1e-4));
}

TEST(MathMatrix, selfadjoint_eigensolver_trivial_double) {
  Eigen::MatrixXd input(3, 3);
  input << 3, 0, 0, 0, 2, 0, 0, 0, 1;
  Eigen::VectorXd sortedDiag(3);
  sortedDiag << 1,2,3;

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXd eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_TRUE(eigenvals.isApprox(sortedDiag));
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXd::Identity(3, 3)));
}

TEST(MathMatrix, selfadjoint_eigensolver_small_double) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXd input = Eigen::MatrixXd::Random(size, size);
  input += input.transpose().eval();

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXd eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_NEAR(input.diagonal().sum(), eigenvals.sum(), 1e-11);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXd::Identity(size, size)));
  EXPECT_TRUE((input * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal()));
}

TEST(MathMatrix, selfadjoint_eigensolver_large_double) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXd input = Eigen::MatrixXd::Random(size, size);
  input += input.transpose().eval();

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXd eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_NEAR(input.diagonal().sum(), eigenvals.sum(), 1e-9);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXd::Identity(size, size), 1e-9));
  EXPECT_TRUE((input * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal()));
}

TEST(MathMatrix, selfadjoint_eigensolver_small_complex) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXcd input = Eigen::MatrixXcd::Random(size, size);
  input += input.adjoint().eval();

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXcd eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_NEAR(input.diagonal().sum().real(), eigenvals.sum(), 1e-11);
  EXPECT_TRUE((eigenvecs * eigenvecs.adjoint()).isApprox(Eigen::MatrixXd::Identity(size, size)));
  EXPECT_TRUE((input * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal()));
}

TEST(MathMatrix, selfadjoint_eigensolver_large_complex) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXcd input = Eigen::MatrixXcd::Random(size, size);
  input += input.adjoint().eval();

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXcd eigenvecs;
  stan::math::selfadjoint_eigensolver(input, eigenvals, eigenvecs);

  EXPECT_NEAR(input.diagonal().sum().real(), eigenvals.sum(), 1e-11);
  EXPECT_TRUE((eigenvecs * eigenvecs.adjoint()).isApprox(Eigen::MatrixXd::Identity(size, size), 1e-8));
  EXPECT_TRUE((input * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal(), 1e-10));
}