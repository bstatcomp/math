#include <stan/math/prim/mat/fun/tridiagonalization.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, tridiagonalization_trivial) {
  Eigen::MatrixXd id = Eigen::MatrixXd::Identity(3, 3);
  Eigen::MatrixXd packed;
  stan::math::internal::block_householder_tridiag(id, packed);
  EXPECT_TRUE(packed.isApprox(id));
  Eigen::MatrixXd q = id;
  stan::math::internal::block_apply_packed_Q(packed, q);
  EXPECT_TRUE(q.isApprox(id));
}

TEST(MathMatrix, tridiagonalization_small) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXd input = Eigen::MatrixXd::Random(size, size);
  input += input.transpose().eval();
  Eigen::MatrixXd packed;

  stan::math::internal::block_householder_tridiag(input, packed);
  Eigen::MatrixXd q = Eigen::MatrixXd::Identity(size, size);
  stan::math::internal::block_apply_packed_Q(packed, q);

  Eigen::MatrixXd t = Eigen::MatrixXd::Constant(size, size, 0);
  t.diagonal() = packed.diagonal();
  t.diagonal(1) = packed.diagonal(1);
  t.diagonal(-1) = packed.diagonal(1);
  EXPECT_TRUE(
      (q * q.transpose()).isApprox(Eigen::MatrixXd::Identity(size, size)));
  EXPECT_TRUE((q * t * q.transpose()).isApprox(input));
}

TEST(MathMatrix, tridiagonalization_large) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXd input = Eigen::MatrixXd::Random(size, size);
  input += input.transpose().eval();
  Eigen::MatrixXd packed;

  stan::math::internal::block_householder_tridiag(input, packed);
  Eigen::MatrixXd q = Eigen::MatrixXd::Identity(size, size);
  stan::math::internal::block_apply_packed_Q(packed, q);

  Eigen::MatrixXd t = Eigen::MatrixXd::Constant(size, size, 0);
  t.diagonal() = packed.diagonal();
  t.diagonal(1) = packed.diagonal(1);
  t.diagonal(-1) = packed.diagonal(1);
  EXPECT_TRUE(
      (q * q.transpose()).isApprox(Eigen::MatrixXd::Identity(size, size)));
  EXPECT_TRUE((q * t * q.transpose()).isApprox(input));
}

TEST(MathMatrix, tridiagonalization_small_complex) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXcd input = Eigen::MatrixXcd::Random(size, size);
  input += input.adjoint().eval();
  Eigen::MatrixXcd packed;

  stan::math::internal::block_householder_tridiag(input, packed);
  Eigen::MatrixXcd q = Eigen::MatrixXcd::Identity(size, size);
  stan::math::internal::block_apply_packed_Q(packed, q);

  Eigen::MatrixXcd t = Eigen::MatrixXcd::Constant(size, size, 0);
  t.diagonal() = packed.diagonal();
  t.diagonal(1) = packed.diagonal(1).conjugate();
  t.diagonal(-1) = packed.diagonal(1);
  EXPECT_TRUE(
      (q * q.adjoint()).isApprox(Eigen::MatrixXcd::Identity(size, size)));
  EXPECT_TRUE((q * t * q.adjoint()).isApprox(input));
}

TEST(MathMatrix, tridiagonalization_big_complex) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::MatrixXcd input = Eigen::MatrixXcd::Random(size, size);
  input += input.adjoint().eval();
  Eigen::MatrixXcd packed;

  stan::math::internal::block_householder_tridiag(input, packed);
  Eigen::MatrixXcd q = Eigen::MatrixXcd::Identity(size, size);
  stan::math::internal::block_apply_packed_Q(packed, q);

  Eigen::MatrixXcd t = Eigen::MatrixXcd::Constant(size, size, 0);
  t.diagonal() = packed.diagonal();
  t.diagonal(1) = packed.diagonal(1).conjugate();
  t.diagonal(-1) = packed.diagonal(1);
  EXPECT_TRUE(
      (q * q.adjoint()).isApprox(Eigen::MatrixXcd::Identity(size, size)));
  EXPECT_TRUE((q * t * q.adjoint()).isApprox(input));
}