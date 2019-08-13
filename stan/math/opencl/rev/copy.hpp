#ifndef STAN_MATH_OPENCL_REV_COPY_HPP
#define STAN_MATH_OPENCL_REV_COPY_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/copy.hpp>
#include <stan/math/opencl/rev/matrix_cl.hpp>
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <type_traits>

namespace stan {
namespace math {

/**
 * Copies the source Eigen matrix to
 * the destination matrix that is stored
 * on the OpenCL device.
 *
 * @tparam R Compile time rows of the Eigen matrix
 * @tparam C Compile time columns of the Eigen matrix
 * @param src source Eigen matrix
 * @return matrix_cl with a copy of the data in the source matrix
 */
template <int R, int C>
inline matrix_cl<var> to_matrix_cl(const Eigen::Matrix<var, R, C>& src) {
  if (src.size() == 0) {
    matrix_cl<var> dst(src.rows(), src.cols());
    return dst;
  }
  matrix_cl<var> dst(src);
  return dst;
}

/**
 * Copies the source matrix that is stored
 * on the OpenCL device to the destination Eigen
 * matrix.
 *
 * @param src source matrix on the OpenCL device
 * @return Eigen matrix with a copy of the data in the source matrix
 */
inline Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> from_matrix_cl(
    const matrix_cl<var>& src) {
  Eigen::Matrix<var, Eigen::Dynamic, Eigen::Dynamic> dst(src.rows(),
                                                           src.cols());
  if (src.size() == 0) {    
    return dst;
  }
  Eigen::MatrixXd vals = from_matrix_cl(src.val_);
  Eigen::MatrixXd adjs = from_matrix_cl(src.adj());
  for (int i = 0; i < dst.size(); i++) {
    std::cout << vals(i) << std::endl;
    dst(i) = var(vals(i));
    dst(i).vi_->adj_ = adjs(i);
  }
  return dst;
}

/**
 * Packs the flat triangular matrix on the OpenCL device and
 * copies it to the std::vector.
 *
 * @param src the flat triangular source matrix on the OpenCL device
 * @return the packed std::vector
 */
inline std::vector<var> packed_copy(const matrix_cl<var>& src) {
  const int packed_size = src.rows() * (src.rows() + 1) / 2;
  std::vector<var> dst(packed_size);
  if (packed_size == 0) {
    return dst;
  }
  std::vector<double> val = packed_copy(src.val());
  std::vector<double> adj = packed_copy(src.adj());
  for (int i = 0; i < packed_size; i++) {
    vari* temp = new vari(val[i]);
    temp->adj_ = adj[i];
    dst[i] = var(temp);
  }
  return dst;
}

/**
 * Copies the packed triangular matrix from
 * the source std::vector to an OpenCL buffer and
 * unpacks it to a flat matrix on the OpenCL device.
 *
 * @tparam partial_view the triangularity of the source matrix
 * @param src the packed source std::vector
 * @param rows the number of rows in the flat matrix
 * @return the destination flat matrix on the OpenCL device
 * @throw <code>std::invalid_argument</code> if the
 * size of the vector does not match the expected size
 * for the packed triangular matrix
 */
template <matrix_cl_view partial_view>
inline matrix_cl<var> packed_copy(const std::vector<var>& src, int rows) {
  const int packed_size = rows * (rows + 1) / 2;
  check_size_match("copy (packed std::vector -> OpenCL)", "src.size()",
                   src.size(), "rows * (rows + 1) / 2", packed_size);
  matrix_cl<var> dst(rows, rows, partial_view);
  if (rows == 0) {
    return dst;
  }
  std::vector<double> val(packed_size);
  std::vector<double> adj(packed_size);

  for (int i = 0; i < packed_size; i++) {
    val[i] = src[i].vi_->val_;
    adj[i] = src[i].vi_->val_;
  }
  dst.val() = packed_copy<partial_view>(val, rows);
  dst.adj() = packed_copy<partial_view>(adj, rows);
  return dst;
}

/**
 * Copies the packed triangular matrix from
 * the source std::vector to an OpenCL buffer and
 * unpacks it to a flat matrix on the OpenCL device.
 *
 * @tparam partial_view the triangularity of the source matrix
 * @param src the packed source std::vector
 * @param rows the number of rows in the flat matrix
 * @return the destination flat matrix on the OpenCL device
 * @throw <code>std::invalid_argument</code> if the
 * size of the vector does not match the expected size
 * for the packed triangular matrix
 */
template <matrix_cl_view partial_view>
inline matrix_cl<var> packed_copy(vari** src, int rows) {
  std::cout << "this?" << std::endl;
  const int packed_size = rows * (rows + 1) / 2;
  matrix_cl<var> dst(rows, rows);
  if (rows == 0) {
    return dst;
  }
  std::vector<double> val(packed_size);
  std::vector<double> adj(packed_size);

  for (int i = 0; i < packed_size; i++) {
    val[i] = src[i]->val_;
    adj[i] = src[i]->adj_;
  }
  dst.val() = packed_copy<partial_view>(val, rows);
  dst.adj() = packed_copy<partial_view>(adj, rows);
  return dst;
}

/**
 * Copies the packed triangular matrix from
 * the source std::vector to an OpenCL buffer and
 * unpacks it to a flat matrix on the OpenCL device.
 *
 * @tparam partial_view the triangularity of the source matrix
 * @param src the packed source std::vector
 * @param rows the number of rows in the flat matrix
 * @return the destination flat matrix on the OpenCL device
 * @throw <code>std::invalid_argument</code> if the
 * size of the vector does not match the expected size
 * for the packed triangular matrix
 */
template <matrix_cl_view partial_view>
vari** packed_copy(const matrix_cl<var>& src) {
  const int packed_size = src.rows() * (src.rows() + 1) / 2;
  vari** varis
      = ChainableStack::instance_->memalloc_.alloc_array<vari*>(packed_size);
  std::vector<var> vec_dst = packed_copy(src);
  for (int i = 0; i < packed_size; i++) {
    varis[i] = vec_dst[i].vi_;
  }
  return varis;
}
/**
 * Copies the source matrix to the
 * destination matrix. Both matrices
 * are stored on the OpenCL device.
 *
 * @param src source matrix
 * @return matrix_cl with copies of values in the source matrix
 * @throw <code>std::invalid_argument</code> if the
 * matrices do not have matching dimensions
 */
inline matrix_cl<var> copy_cl(const matrix_cl<var>& src) {
  matrix_cl<var> dst(src.rows(), src.cols());
  if (src.size() == 0) {
    return dst;
  }
  dst.val() = src.val();
  dst.adj() = src.adj();
  return dst;
}

}  // namespace math
}  // namespace stan
#endif
#endif
