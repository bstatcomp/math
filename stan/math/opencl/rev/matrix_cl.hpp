#ifndef STAN_MATH_OPENCL_REV_MATRIX_CL_HPP
#define STAN_MATH_OPENCL_REV_MATRIX_CL_HPP
#ifdef STAN_OPENCL
#include <stan/math/opencl/constants.hpp>
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/buffer_types.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/arr/fun/vec_concat.hpp>
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

template <typename T>
class matrix_cl<T, enable_if_var_or_vari<T>> {
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
  mutable TriangularViewCL triangular_view_;
 public:
  int rows() const { return rows_; }

  int cols() const { return cols_; }

  int size() const { return rows_ * cols_; }
  matrix_cl<double>& val() const {return val_;}
  matrix_cl<double>& adj() const {return adj_;}
  explicit matrix_cl() : rows_(0), cols_(0) {}

  template <TriangularViewCL triangular_view = TriangularViewCL::Entire, typename = enable_if_var_or_vari<T>>
  void zeros();
  template <TriangularMapCL triangular_map = TriangularMapCL::LowerToUpper, typename = enable_if_var_or_vari<T>>
  void triangular_transpose();
  template <TriangularViewCL triangular_view = TriangularViewCL::Entire, typename = enable_if_var_or_vari<T>>
  void sub_block(const matrix_cl<T, enable_if_var_or_vari<T>>& A, size_t A_i, size_t A_j, size_t this_i,
                 size_t this_j, size_t nrows, size_t ncols);

  template <int R, int C>
  explicit matrix_cl(const Eigen::Matrix<T, R, C>& A)
      : rows_(A.rows()), cols_(A.cols()),
      val_(A.val().eval()),
      adj_(A.adj().eval()) {}


  explicit matrix_cl(vari** A, const int& R, const int& C) :
    rows_(R), cols_(C), adj_(Eigen::Map<const matrix_vi>(A, R, C).adj().eval()),
    val_(Eigen::Map<const matrix_vi>(A, R, C).val().eval()) {
  }

  explicit matrix_cl(const int& rows, const int& cols) :
  rows_(rows), cols_(cols), val_(rows, cols), adj_(rows, cols) {}

  matrix_cl<T> operator=(const matrix_cl<T>& A) {
    val_ = A.val();
    adj_ = A.adj();
    return *this;
  }
};

}
}

#endif
#endif
