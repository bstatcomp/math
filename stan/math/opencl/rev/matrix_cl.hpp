#ifndef STAN_MATH_OPENCL_REV_MATRIX_CL_HPP
#define STAN_MATH_OPENCL_REV_MATRIX_CL_HPP
#ifdef STAN_OPENCL

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/rev/scal/meta/enable_if_var_or_vari.hpp>
#include <CL/cl.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace stan {
namespace math {

template <>
class matrix_cl<var> {
 private:
  /**
   * cl::Buffer provides functionality for working with the OpenCL buffer.
   * An OpenCL buffer allocates the memory in the device that
   * is provided by the context.
   */
  const int rows_;
  const int cols_;
  mutable matrix_cl<double> val_;
  mutable matrix_cl<double> adj_;
  matrix_cl_view view_;  // Holds info on if matrix is a special type
 public:
  typedef var type;
  // Forward declare the methods that work in place on the matrix
  template <matrix_cl_view matrix_view = matrix_cl_view::Entire>
  void zeros();
  template <TriangularMapCL triangular_map = TriangularMapCL::LowerToUpper>
  void triangular_transpose();
  void sub_block(const matrix_cl<var>& A, size_t A_i, size_t A_j, size_t this_i,
                 size_t this_j, size_t nrows, size_t ncols);

  int rows() const { return this->rows_; }

  int cols() const { return this->cols_; }

  int size() const { return this->rows_ * this->cols_; }
  matrix_cl<double>& val() const { return this->val_; }
  matrix_cl<double>& adj() const { return this->adj_; }

  const matrix_cl_view& view() const { return view_; }
  void view(const matrix_cl_view& view) {
    view_ = view;
    this->val_.view(view);
    this->adj_.view(view);
  }

  explicit matrix_cl() : rows_(0), cols_(0) {}

  matrix_cl(const matrix_cl<var>& A)
      : rows_(A.rows()),
        cols_(A.cols()),
        view_(A.view()),
        val_(A.val()),
        adj_(A.adj()) {}

  template <int R, int C>
  explicit matrix_cl(const Eigen::Matrix<var, R, C>& A,
                     matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(A.rows()), cols_(A.cols()), view_(partial_view),
        val_(A.val().eval(), partial_view), adj_(A.adj().eval(), partial_view) {}

  template <int R, int C>
  explicit matrix_cl(const Eigen::Matrix<vari*, R, C>& A,
                     matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(A.rows()),
        cols_(A.cols()),
        view_(partial_view),
        val_(A.val().eval(), partial_view),
        adj_(A.adj().eval(), partial_view) {}

  explicit matrix_cl(vari** A, const int& R, const int& C,
                     matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(R),
        cols_(C),
        view_(partial_view),
        adj_(Eigen::Map<matrix_vi>(A, R, C).adj().eval(), partial_view),
        val_(Eigen::Map<matrix_vi>(A, R, C).val().eval(), partial_view) {}

  matrix_cl(const int& rows, const int& cols,
            matrix_cl_view partial_view = matrix_cl_view::Entire)
      : rows_(rows),
        cols_(cols),
        view_(partial_view),
        val_(rows, cols, partial_view),
        adj_(rows, cols, partial_view) {}

  matrix_cl<var> operator=(const matrix_cl<var>& A) {
    check_size_match("assignment of (OpenCL) matrices", "source.rows()",
                     A.rows(), "destination.rows()", this->rows());
    check_size_match("assignment of (OpenCL) matrices", "source.cols()",
                     A.cols(), "destination.cols()", this->cols());
    val_ = A.val();
    adj_ = A.adj();
    view_ = A.view();
    return *this;
  }
};

}  // namespace math
}  // namespace stan

#endif
#endif
