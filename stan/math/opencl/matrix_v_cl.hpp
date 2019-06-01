#ifndef STAN_MATH_OPENCL_MATRIX_V_CL_HPP
#define STAN_MATH_OPENCL_MATRIX_V_CL_HPP
#ifdef STAN_OPENCL
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/constants.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/opencl/cache_copy.hpp>
#include <stan/math/opencl/kernels/pack.hpp>
#include <stan/math/opencl/kernels/unpack.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/arr/fun/vec_concat.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <CL/cl.hpp>
#include <string>
#include <vector>
#include <algorithm>

namespace stan {
namespace math {
/**
 * Represents a matrix of varis on the OpenCL device.
 *
 * The matrix data is stored in two separate matrix_cl
 * members for values (val_) and adjoints (adj_).
 */
template <TriangularViewCL triangular_view = TriangularViewCL::Entire>
class matrix_v_cl {
 private:
  const int rows_;
  const int cols_;

 public:
  matrix_cl val_;
  matrix_cl adj_;
  int rows() const { return rows_; }

  int cols() const { return cols_; }

  int size() const { return rows_ * cols_; }

  matrix_v_cl() : rows_(0), cols_(0) {}

  /**
   * Constructor for the matrix_v_cl
   * for triangular matrices.
   *
   *
   * @param A the Eigen matrix
   * @param M The Rows and Columns of the matrix
   *
   * @throw <code>std::system_error</code> if the
   * matrices do not have matching dimensions
   */
  matrix_v_cl(vari**& A, int M) : rows_(M), cols_(M), val_(M, M), adj_(M, M) {
    if (size() == 0)
      return;
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    try {
      const int vari_size = M * (M + 1) / 2;
      std::vector<double> val_cpy(vari_size);
      std::vector<double> adj_cpy(vari_size);
      for (size_t j = 0; j < vari_size; ++j) {
        val_cpy[j] = A[j]->val_;
        adj_cpy[j] = A[j]->adj_;
      }
      matrix_cl packed_val(val_cpy);
      matrix_cl packed_adj(adj_cpy);
      queue.enqueueWriteBuffer(packed_val.buffer(), CL_TRUE, 0,
                               sizeof(double) * vari_size, val_cpy.data());
      queue.enqueueWriteBuffer(packed_adj.buffer(), CL_TRUE, 0,
                               sizeof(double) * vari_size, adj_cpy.data());
      stan::math::opencl_kernels::unpack(cl::NDRange(M, M), val_,
                                         packed_val, M, M,
                                         triangular_view);
      stan::math::opencl_kernels::unpack(cl::NDRange(M, M), adj_,
                                         packed_adj, M, M,
                                         triangular_view);
    } catch (const cl::Error& e) {
      check_opencl_error("matrix constructor", e);
    }
  }
};

// if the triangular view is entire dont unpack, just copy
template <>
inline matrix_v_cl<TriangularViewCL::Entire>::matrix_v_cl(vari**& A, int M)
    : rows_(M), cols_(M), val_(M, M), adj_(M, M) {
  if (size() == 0)
    return;
  const int vari_size = M * M;
  std::vector<double> val_cpy(vari_size);
  std::vector<double> adj_cpy(vari_size);
  for (size_t j = 0; j < vari_size; ++j) {
    val_cpy[j] = A[j]->val_;
    adj_cpy[j] = A[j]->adj_;
  }
  val_ = matrix_cl(val_cpy, M, M);
  val_ = matrix_cl(adj_cpy, M, M);
}

}  // namespace math
}  // namespace stan

#endif
#endif
