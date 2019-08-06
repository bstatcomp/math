#ifdef STAN_OPENCL
#include <stan/math/rev/mat.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <stan/math/opencl/rev/copy.hpp>
#include <gtest/gtest.h>
#include <CL/cl.hpp>
#include <algorithm>
#include <vector>

TEST(MathMatrixGPU, matrix_cl_vector_copy) {
  using stan::math::vector_v;
  using stan::math::var;
  using stan::math::matrix_cl;
  vector_v d1_cpu;
  vector_v d1_a_cpu;
  vector_v d1_b_cpu;
  d1_cpu.resize(3);
  d1_a_cpu.resize(3);
  d1_b_cpu.resize(3);
  d1_cpu << 1, 2, 3;
  // vector
  matrix_cl<var> d11_cl(3, 1);
  matrix_cl<var> d111_cl(3, 1);
  EXPECT_NO_THROW(d11_cl = to_matrix_cl(d1_cpu));
  EXPECT_NO_THROW(d111_cl = copy_cl(d11_cl));
  EXPECT_NO_THROW(d1_a_cpu = from_matrix_cl(d11_cl));
  EXPECT_NO_THROW(d1_b_cpu = from_matrix_cl(d111_cl));
  EXPECT_EQ(1, d1_a_cpu(0).vi_->val_);
  EXPECT_EQ(2, d1_a_cpu(1).vi_->val_);
  EXPECT_EQ(3, d1_a_cpu(2).vi_->val_);
  EXPECT_EQ(1, d1_b_cpu(0).vi_->val_);
  EXPECT_EQ(2, d1_b_cpu(1).vi_->val_);
  EXPECT_EQ(3, d1_b_cpu(2).vi_->val_);
}

TEST(MathMatrixCL, matrix_cl_matrix_copy) {
  using stan::math::matrix_v;
  using stan::math::matrix_cl;
  using stan::math::var;
  matrix_v d2_cpu;
  matrix_v d2_a_cpu;
  matrix_v d2_b_cpu;
  matrix_v d0_cpu;
  d2_cpu.resize(2, 3);
  d2_a_cpu.resize(2, 3);
  d2_b_cpu.resize(2, 3);
  d2_cpu << 1, 2, 3, 4, 5, 6;
  // matrix
  matrix_cl<var> d00_cl;
  matrix_cl<var> d000_cl;
  matrix_cl<var> d22_cl(2, 3);
  matrix_cl<var> d222_cl(2, 3);
  EXPECT_NO_THROW(d22_cl = to_matrix_cl(d2_cpu));
  EXPECT_NO_THROW(d222_cl = copy_cl(d22_cl));
  EXPECT_NO_THROW(d2_a_cpu = from_matrix_cl(d22_cl));
  EXPECT_NO_THROW(d2_b_cpu = from_matrix_cl(d222_cl));
  EXPECT_EQ(1, d2_a_cpu(0, 0).vi_->val_);
  EXPECT_EQ(2, d2_a_cpu(0, 1).vi_->val_);
  EXPECT_EQ(3, d2_a_cpu(0, 2).vi_->val_);
  EXPECT_EQ(4, d2_a_cpu(1, 0).vi_->val_);
  EXPECT_EQ(5, d2_a_cpu(1, 1).vi_->val_);
  EXPECT_EQ(6, d2_a_cpu(1, 2).vi_->val_);
  EXPECT_EQ(1, d2_b_cpu(0, 0).vi_->val_);
  EXPECT_EQ(3, d2_b_cpu(0, 2).vi_->val_);
  EXPECT_EQ(4, d2_b_cpu(1, 0).vi_->val_);
  EXPECT_EQ(2, d2_b_cpu(0, 1).vi_->val_);
  EXPECT_EQ(5, d2_b_cpu(1, 1).vi_->val_);
  EXPECT_EQ(6, d2_b_cpu(1, 2).vi_->val_);
  // zero sized copy
  EXPECT_NO_THROW(d00_cl = to_matrix_cl(d0_cpu));
  EXPECT_NO_THROW(d0_cpu = from_matrix_cl(d00_cl));
  EXPECT_NO_THROW(d000_cl = copy_cl(d00_cl));
}
/*
TEST(MathMatrixCL, matrix_cl_pack_unpack_copy_lower) {
  using stan::math::matrix_v;
  using stan::math::matrix_d;
  using stan::math::matrix_cl;
  using stan::math::var;
  using stan::math::vari;
  using stan::math::matrix_cl_view;
  int size = 42;
  int packed_size = size * (size + 1) / 2;
  vari** packed_vari(stan::math::ChainableStack::instance_->memalloc_.alloc_array<vari*>(packed_size));
  std::vector<var> packed_vari_dst(packed_size);
  for (size_t i = 0; i < packed_size; i++) {
    packed_vari[i] = new vari(i, false);
    packed_vari_dst[i].vi_ = new vari(0, false);
  }
  matrix_cl<var> m_cl = stan::math::packed_copy<matrix_cl_view::Lower>(packed_vari, size);
  stan::math::packed_copy<matrix_cl_view::Lower>(m_cl, packed_vari_dst.vi_);
  size_t pos = 0;
  for (size_t j = 0; j < m_cl.cols(); ++j) {
    for (size_t i = 0; i < j; i++) {
      EXPECT_EQ(packed_vari_dst[i]->val_, packed_vari[i]->val_);
      EXPECT_EQ(packed_vari_dst[i]->adj_, packed_vari[i]->adj_);
      ++pos;
    }
  }
}
  packed_mat_dst
      = packed_copy<matrix_cl_view::Lower>(m_cl);
  for (size_t i = 0; i < packed_mat.size(); i++) {
    EXPECT_EQ(packed_mat[i], packed_mat_dst[i]);
  }
*/

/*
TEST(MathMatrixCL, matrix_cl_pack_unpack_copy_upper) {
  using stan::math::matrix_v;
  using stan::math::matrix_cl;
  int size = 51;
  int packed_size = size * (size + 1) / 2;
  std::vector<var> packed_mat(packed_size);
  std::vector<var> packed_mat_dst(packed_size);
  for (size_t i = 0; i < packed_mat.size(); i++) {
    packed_mat[i] = i;
  }
  matrix_d m_flat_cpu(size, size);
  auto m_cl = packed_copy<matrix_cl_view::Upper>(
      packed_mat, size);
  m_flat_cpu = from_matrix_cl(m_cl);
  size_t pos = 0;
  for (size_t j = 0; j < size; ++j) {
    for (size_t i = 0; i <= j; i++) {
      EXPECT_EQ(m_flat_cpu(i, j), packed_mat[pos]);
      pos++;
    }
    for (size_t i = j + 1; i < size; ++i) {
      EXPECT_EQ(m_flat_cpu(i, j), 0.0);
    }
  }
  packed_mat_dst
      = packed_copy<matrix_cl_view::Upper>(m_cl);
  for (size_t i = 0; i < packed_mat.size(); i++) {
    EXPECT_EQ(packed_mat[i], packed_mat_dst[i]);
  }
}

TEST(MathMatrixCL, matrix_cl_pack_unpack_copy_exception) {
  using stan::math::matrix_cl;
  std::vector<var> packed_mat;
  matrix_cl<var> m_cl_zero;
  EXPECT_NO_THROW(packed_copy<matrix_cl_view::Upper>(
      packed_mat, 0));
  EXPECT_NO_THROW(
      packed_copy<matrix_cl_view::Upper>(m_cl_zero));
  EXPECT_THROW(packed_copy<matrix_cl_view::Upper>(
                   packed_mat, 1),
               std::invalid_argument);
}
*/
#endif
