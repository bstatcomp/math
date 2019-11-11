#include <stan/math/prim/mat/fun/mrrr.hpp>
#include <gtest/gtest.h>
#include <algorithm>

TEST(MathMatrix, tridiag_eigensolver_trivial_float) {
  Eigen::VectorXf diag(3), subdiag(2), eigenvals;
  diag << 1.5f, 1.2f, -2.f;
  subdiag << 0.f, 0.f;

  Eigen::MatrixXf eigenvecs(3, 3);
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals,eigenvecs);
  std::sort(diag.data(), diag.data()+3);
  EXPECT_TRUE(eigenvals.isApprox(diag));
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXf::Identity(3, 3)));
}

TEST(MathMatrix, tridiag_eigensolver_small_float) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::VectorXf diag = Eigen::VectorXf::Random(size);
  Eigen::VectorXf subdiag = Eigen::VectorXf::Random(size - 1);
  subdiag[2] = 0;

  Eigen::VectorXf eigenvals;
  Eigen::MatrixXf eigenvecs;
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals,eigenvecs);

  Eigen::MatrixXf t = Eigen::MatrixXf::Constant(size, size, 0);
  t.diagonal() = diag;
  t.diagonal(1) = subdiag;
  t.diagonal(-1) = subdiag;

  EXPECT_NEAR(diag.sum(), eigenvals.sum(), 1e-6);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXf::Identity(size, size)));
  EXPECT_TRUE((t * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal()));
}

TEST(MathMatrix, tridiag_eigensolver_large_float) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::VectorXf diag = Eigen::VectorXf::Random(size);
  Eigen::VectorXf subdiag = Eigen::VectorXf::Random(size - 1);
  subdiag[12] = 0;
  subdiag[120] = 0;
  subdiag[121] = 0;

  Eigen::VectorXf eigenvals;
  Eigen::MatrixXf eigenvecs;
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals,eigenvecs);

  Eigen::MatrixXf t = Eigen::MatrixXf::Constant(size, size, 0);
  t.diagonal() = diag;
  t.diagonal(1) = subdiag;
  t.diagonal(-1) = subdiag;

  EXPECT_NEAR(diag.sum(), eigenvals.sum(), 1e-5);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXf::Identity(size, size), 1e-5));
  EXPECT_TRUE((t * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal(), 1e-5));
}

TEST(MathMatrix, tridiag_eigensolver_trivial_double) {
  Eigen::VectorXd diag(3), subdiag(2), eigenvals;
  diag << 1.5, 1.2, -2;
  subdiag << 0, 0;

  Eigen::MatrixXd eigenvecs(3, 3);
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals,eigenvecs);
  std::sort(diag.data(), diag.data()+3);
  EXPECT_TRUE(eigenvals.isApprox(diag));
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXd::Identity(3, 3)));
}

TEST(MathMatrix, tridiag_eigensolver_small_double) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::VectorXd diag = Eigen::VectorXd::Random(size);
  Eigen::VectorXd subdiag = Eigen::VectorXd::Random(size - 1);
  subdiag[2] = 0;

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXd eigenvecs;
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals,eigenvecs);

  Eigen::MatrixXd t = Eigen::MatrixXd::Constant(size, size, 0);
  t.diagonal() = diag;
  t.diagonal(1) = subdiag;
  t.diagonal(-1) = subdiag;

  EXPECT_NEAR(diag.sum(), eigenvals.sum(), 1e-11);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose()).isApprox(Eigen::MatrixXd::Identity(size, size)));
  EXPECT_TRUE((t * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal()));
}

TEST(MathMatrix, tridiag_eigensolver_large_double) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::VectorXd diag = Eigen::VectorXd::Random(size);
  Eigen::VectorXd subdiag = Eigen::VectorXd::Random(size - 1);
  subdiag[12] = 0;
  subdiag[120] = 0;
  subdiag[121] = 0;

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXd eigenvecs;
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals,eigenvecs);

  Eigen::MatrixXd t = Eigen::MatrixXd::Constant(size, size, 0);
  t.diagonal() = diag;
  t.diagonal(1) = subdiag;
  t.diagonal(-1) = subdiag;

  EXPECT_NEAR(diag.sum(), eigenvals.sum(), 1e-9);
  EXPECT_TRUE((eigenvecs * eigenvecs.transpose())
                  .isApprox(Eigen::MatrixXd::Identity(size, size), 1e-9));
  EXPECT_TRUE((t * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal(), 1e-9));
}

TEST(MathMatrix, tridiag_eigensolver_small_complex) {
  int size = 7;
  srand(0);  // ensure test repeatability
  Eigen::VectorXd diag = Eigen::VectorXd::Random(size);
  Eigen::VectorXcd subdiag = Eigen::VectorXcd::Random(size - 1);
  subdiag[2] = 0.;

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXcd eigenvecs;
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals,eigenvecs);
  stan::math::internal::sort_eigenpairs(eigenvals, eigenvecs);

  Eigen::MatrixXcd t = Eigen::MatrixXcd::Constant(size, size, 0);
  t.diagonal() = diag;
  t.diagonal(1) = subdiag.conjugate();
  t.diagonal(-1) = subdiag;

  EXPECT_NEAR(diag.sum(), eigenvals.sum(), 1e-11);
  EXPECT_TRUE((eigenvecs * eigenvecs.adjoint()).isApprox(Eigen::MatrixXcd::Identity(size, size)));
  EXPECT_TRUE((t * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal(), 1e-14));
}

TEST(MathMatrix, tridiag_eigensolver_large_complex) {
  int size = 345;
  srand(0);  // ensure test repeatability
  Eigen::VectorXd diag = Eigen::VectorXd::Random(size);
  Eigen::VectorXcd subdiag = Eigen::VectorXd::Random(size - 1);
  subdiag[12] = 0;
  subdiag[120] = 0;
  subdiag[121] = 0;

  Eigen::VectorXd eigenvals;
  Eigen::MatrixXcd eigenvecs;
  stan::math::internal::tridiagonal_eigensolver(diag, subdiag, eigenvals, eigenvecs);

  Eigen::MatrixXcd t = Eigen::MatrixXcd::Constant(size, size, 0);
  t.diagonal() = diag;
  t.diagonal(1) = subdiag.conjugate();
  t.diagonal(-1) = subdiag;

  EXPECT_NEAR(diag.sum(), eigenvals.sum(), 1e-9);
  EXPECT_TRUE((eigenvecs * eigenvecs.adjoint()).isApprox(Eigen::MatrixXd::Identity(size, size), 1e-9));
  EXPECT_TRUE((t * eigenvecs).isApprox(eigenvecs * eigenvals.asDiagonal(), 1e-9));
}