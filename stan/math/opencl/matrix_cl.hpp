#ifndef STAN_MATH_OPENCL_MATRIX_CL_HPP
#define STAN_MATH_OPENCL_MATRIX_CL_HPP
#ifdef STAN_OPENCL
#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/constants.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/opencl/cache_copy.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/arr/fun/vec_concat.hpp>
#include <stan/math/rev/scal/fun/value_of_rec.hpp>
#include <CL/cl.hpp>
#include <string>
#include <vector>
#include <algorithm>
#include <type_traits>

/**
 *  @file stan/math/opencl/matrix_cl.hpp
 *  @brief The matrix_cl class - allocates memory space on the OpenCL device,
 *    functions for transfering matrices to and from OpenCL devices
 */
namespace stan {
namespace math {
/**
 * Represents a matrix on the OpenCL device.
 *
 * The matrix data is stored in the oclBuffer_.
 */
class matrix_cl {
 private:
  /**
   * cl::Buffer provides functionality for working with the OpenCL buffer.
   * An OpenCL buffer allocates the memory in the device that
   * is provided by the context.
   */
  cl::Buffer oclBuffer_;
  const int rows_;
  const int cols_;
  mutable std::vector<cl::Event> write_events_;  // Tracks write jobs
  mutable std::vector<cl::Event> read_events_;   // Tracks reads

 public:
  // Forward declare the methods that work in place on the matrix
  template <TriangularViewCL triangular_view = TriangularViewCL::Entire>
  void zeros();
  template <TriangularMapCL triangular_map = TriangularMapCL::LowerToUpper>
  void triangular_transpose();
  template <TriangularViewCL triangular_view = TriangularViewCL::Entire>
  void sub_block(const matrix_cl& A, size_t A_i, size_t A_j, size_t this_i,
                 size_t this_j, size_t nrows, size_t ncols);
  int rows() const { return rows_; }

  int cols() const { return cols_; }

  int size() const { return rows_ * cols_; }

  /**
   * Clear the write events from the event stacks.
   */
  inline void clear_write_events() const {
    write_events_.clear();
    return;
  }

  /**
   * Clear the read events from the event stacks.
   */
  inline void clear_read_events() const {
    read_events_.clear();
    return;
  }

  /**
   * Clear the write events from the event stacks.
   */
  inline void clear_read_write_events() const {
    read_events_.clear();
    write_events_.clear();
    return;
  }

  /**
   * Get the events from the event stacks.
   * @return The write event stack.
   */
  inline const std::vector<cl::Event>& write_events() const {
    return write_events_;
  }

  /**
   * Get the events from the event stacks.
   * @return The read/write event stack.
   */
  inline const std::vector<cl::Event>& read_events() const {
    return read_events_;
  }

  /**
   * Get the events from the event stacks.
   * @return The read/write event stack.
   */
  inline const std::vector<cl::Event> read_write_events() const {
    return vec_concat(this->read_events(), this->write_events());
  }

  /**
   * Add an event to the read event stack.
   * @param new_event The event to be pushed on the event stack.
   */
  inline void add_read_event(cl::Event new_event) const {
    this->read_events_.push_back(new_event);
  }

  /**
   * Add an event to the write event stack.
   * @param new_event The event to be pushed on the event stack.
   */
  inline void add_write_event(cl::Event new_event) const {
    this->write_events_.push_back(new_event);
  }

  /**
   * Add an event to the read/write event stack.
   * @param new_event The event to be pushed on the event stack.
   */
  inline void add_read_write_event(cl::Event new_event) const {
    this->read_events_.push_back(new_event);
    this->write_events_.push_back(new_event);
  }

  /**
   * Waits for the write events and clears the read event stack.
   */
  inline void wait_for_write_events() const {
    cl::CommandQueue queue = opencl_context.queue();
    cl::Event copy_event;
    queue.enqueueBarrierWithWaitList(&this->write_events(), &copy_event);
    copy_event.wait();
    write_events_.clear();
    return;
  }

  /**
   * Waits for the read events and clears the read event stack.
   */
  inline void wait_for_read_events() const {
    cl::CommandQueue queue = opencl_context.queue();
    cl::Event copy_event;
    queue.enqueueBarrierWithWaitList(&this->read_events(), &copy_event);
    copy_event.wait();
    read_events_.clear();
    return;
  }

  /**
   * Waits for read and write events to finish and clears the read, write, and
   * read/write event stacks.
   */
  inline void wait_for_read_write_events() const {
    cl::CommandQueue queue = opencl_context.queue();
    cl::Event copy_event;
    const std::vector<cl::Event> mat_events = this->read_write_events();
    queue.enqueueBarrierWithWaitList(&mat_events, &copy_event);
    copy_event.wait();
    read_events_.clear();
    write_events_.clear();
    return;
  }

  const cl::Buffer& buffer() const { return oclBuffer_; }
  matrix_cl() : rows_(0), cols_(0) {}

  matrix_cl(cl::Buffer& A, const int R, const int C) : rows_(R), cols_(C) {
    oclBuffer_ = A;
  }
  template <typename T>
  explicit matrix_cl(std::vector<T> A) : rows_(1), cols_(A.size()) {
    if (A.size() == 0)
      return;
    // the context is needed to create the buffer object
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    try {
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * A.size());
      queue.enqueueWriteBuffer(oclBuffer_, CL_TRUE, 0,
                               sizeof(double) * A.size(),
                               value_of_rec(A).data());
    } catch (const cl::Error& e) {
      check_opencl_error("matrix_cl(std::vector<T>) constructor", e);
    }
  }

  explicit matrix_cl(std::vector<double> A) : rows_(1), cols_(A.size()) {
    if (A.size() == 0)
      return;
    // the context is needed to create the buffer object
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    try {
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * A.size());
      queue.enqueueWriteBuffer(oclBuffer_, CL_TRUE, 0,
                               sizeof(double) * A.size(), A.data());
    } catch (const cl::Error& e) {
      check_opencl_error("matrix_cl(std::vector<T>) constructor", e);
    }
  }

  template <typename T>
  explicit matrix_cl(std::vector<T> A, const int R, const int C)
      : rows_(R), cols_(C) {
    if (A.size() == 0)
      return;
    // the context is needed to create the buffer object
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    try {
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * A.size());
      queue.enqueueWriteBuffer(oclBuffer_, CL_TRUE, 0,
                               sizeof(double) * A.size(),
                               value_of_rec(A).data());
    } catch (const cl::Error& e) {
      check_opencl_error("matrix_cl(std::vector<T>, rows, cols) constructor",
                         e);
    }
  }

  explicit matrix_cl(std::vector<double> A, const int R, const int C)
      : rows_(R), cols_(C) {
    if (A.size() == 0)
      return;
    // the context is needed to create the buffer object
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    try {
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * A.size());
      queue.enqueueWriteBuffer(oclBuffer_, CL_TRUE, 0,
                               sizeof(double) * A.size(), A.data());
    } catch (const cl::Error& e) {
      check_opencl_error("matrix_cl(std::vector<T>, rows, cols) constructor",
                         e);
    }
  }

  matrix_cl(const matrix_cl& A) : rows_(A.rows()), cols_(A.cols()) {
    if (A.size() == 0)
      return;
    // the context is needed to create the buffer object
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue queue = opencl_context.queue();
    try {
      // creates a read&write object for "size" double values
      // in the provided context
      oclBuffer_ = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * size());
      cl::Event cstr_event;
      queue.enqueueCopyBuffer(A.buffer(), this->buffer(), 0, 0,
                              A.size() * sizeof(double), &A.write_events(),
                              &cstr_event);
      this->add_write_event(cstr_event);
    } catch (const cl::Error& e) {
      check_opencl_error("copy (OpenCL)->(OpenCL)", e);
    }
  }
  /**
   * Constructor for the matrix_cl that
   * only allocates the buffer on the OpenCL device.
   *
   * @param rows number of matrix rows, must be greater or equal to 0
   * @param cols number of matrix columns, must be greater or equal to 0
   *
   * @throw <code>std::system_error</code> if the
   * matrices do not have matching dimensions
   *
   */
  matrix_cl(const int& rows, const int& cols) : rows_(rows), cols_(cols) {
    if (size() == 0) {
      return;
    }
    cl::Context& ctx = opencl_context.context();
    try {
      // creates the OpenCL buffer of the provided size
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * rows_ * cols_);
    } catch (const cl::Error& e) {
      check_opencl_error("matrix constructor", e);
    }
  }

  /**
   * Constructor for the matrix_cl that
   * creates a copy of the Eigen matrix on the OpenCL device. If the matrix contains integers they will be converted to doubles.
   *
   * @tparam T type of matrix
   * @tparam R rows of matrix
   * @tparam C cols of matrix
   * @param A Eigen matrix
   *
   * @throw <code>std::system_error</code> if the
   * matrices do not have matching dimensions
   */
  template <typename T, int R, int C, typename Cond = std::enable_if<std::is_same<T,double>::value || std::is_same<T,int>::value>>
  explicit matrix_cl(const Eigen::Matrix<T, R, C>& A)
      : rows_(A.rows()), cols_(A.cols()) {
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    if (A.size() > 0) {
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * A.size());
      internal::cache_copy(oclBuffer_, A);
    }
  }

  /**
   * Constructor for the matrix_cl that
   * creates a copy of the Eigen matrix on the OpenCL device. If the matrix contains integers they will be converted to doubles.
   *
   *
   * @tparam T type of data in the Eigen matrix
   * @param A the Eigen matrix
   *
   * @throw <code>std::system_error</code> if the
   * matrices do not have matching dimensions
   */
  template <typename T, int R, int C, typename Cond = std::enable_if<std::is_same<T,double>::value || std::is_same<T,int>::value>>
  explicit matrix_cl(const Eigen::Map<const Eigen::Matrix<T, R, C>>& A)
          : rows_(A.rows()), cols_(A.cols()) {
    if (size() > 0) {
      cl::Context& ctx = opencl_context.context();
      cl::CommandQueue& queue = opencl_context.queue();
      try {
        // creates the OpenCL buffer to copy the Eigen
        // matrix to the OpenCL device
        oclBuffer_
                = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * A.size());
        /**
         * Writes the contents of A to the OpenCL buffer
         * starting at the offset 0.
         * CL_TRUE denotes that the call is blocking as
         * we do not want to execute any further kernels
         * on the device until we are sure that the data
         * is finished transfering)
         */
        queue.enqueueWriteBuffer(oclBuffer_, CL_TRUE, 0,
                                 sizeof(double) * A.size(), A.data());
      } catch (const cl::Error& e) {
        check_opencl_error("matrix constructor", e);
      }
    }
  }

    /**
   * Constructor for the matrix_cl that contains a single value.
   *
   * @param A the value
   *
   * @throw <code>std::system_error</code> if the
   * matrices do not have matching dimensions
   */
    explicit matrix_cl(double A)
            : rows_(1), cols_(1) {
      cl::Context& ctx = opencl_context.context();
      cl::CommandQueue& queue = opencl_context.queue();
      try {
        // creates the OpenCL buffer to copy the Eigen
        // matrix to the OpenCL device
        oclBuffer_
                = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double));
        /**
         * Writes the contents of A to the OpenCL buffer
         * starting at the offset 0.
         * CL_TRUE denotes that the call is blocking as
         * we do not want to execute any further kernels
         * on the device until we are sure that the data
         * is finished transfering)
         */
        queue.enqueueWriteBuffer(oclBuffer_, CL_TRUE, 0, sizeof(double), &A);
      } catch (const cl::Error& e) {
        check_opencl_error("matrix constructor", e);
      }
    }

    /**
   * Constructs a const matrix_cl that constains a copy of the Eigen matrix on the OpenCL device. If the matrix contains integers they will be converted to doubles.
   * If the matrix already has a cached copy on the device, the cache is used and no copying is done.
   *
   *
   * @tparam R row type of input matrix
   * @tparam C column type of input matrix
   * @param A the Eigen matrix
   */
    template <typename T, int R, int C, typename Cond = std::enable_if<std::is_same<T,double>::value || std::is_same<T,int>::value>>
    static const matrix_cl constant(const Eigen::Matrix<T, R, C>& A){
#ifdef STAN_OPENCL_CACHE
      if (A.opencl_buffer_() != NULL) {
        return matrix_cl(A.opencl_buffer_, A.rows(), A.cols());
      }
      else{
        return matrix_cl(A);
      }
#else
      return matrix_cl(A);
#endif
    }

    /**
      * Constructs a const matrix_cl that constains a single value on the OpenCL device.
      *
      *
      * @tparam R row type of input matrix
      * @tparam C column type of input matrix
      * @param A the Eigen matrix
      */
      template<typename T>
    static const matrix_cl constant(T A){
      return matrix_cl(A);
    }

  /**
   * Constructor for the matrix_cl that
   * creates a copy of a var type Eigen matrix on the GPU.
   *
   * @tparam R rows of matrix
   * @tparam C cols of matrix
   * @param A the Eigen matrix
   */
  template <int R, int C>
  explicit matrix_cl(const Eigen::Matrix<var, R, C>& A)
      : rows_(A.rows()), cols_(A.cols()) {
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    if (A.size() > 0) {
      Eigen::Matrix<double, -1, -1> L_A(value_of_rec(A));
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * L_A.size());
      queue.enqueueWriteBuffer(oclBuffer_, CL_TRUE, 0,
                               sizeof(double) * L_A.size(), L_A.data());
    }
  }
  /**
   * Constructor for the matrix_cl that
   * creates a copy of the double array representing a matrix
   * on the OpenCL devioce.
   *
   * @param A the input double array
   *
   * @throw <code>std::system_error</code> if the
   * matrices do not have matching dimensions
   */
  explicit matrix_cl(double *A, int rows, int cols)
      : rows_(rows), cols_(cols) {
    if (size() == 0) {
      return;
    }
    cl::Context& ctx = opencl_context.context();
    cl::CommandQueue& queue = opencl_context.queue();
    try {
      // creates the OpenCL buffer to copy the Eigen
      // matrix to the OpenCL device
      oclBuffer_
          = cl::Buffer(ctx, CL_MEM_READ_WRITE, sizeof(double) * rows * cols);
      /**
       * Writes the contents of A to the OpenCL buffer
       * starting at the offset 0.
       * CL_TRUE denotes that the call is blocking as
       * we do not want to execute any further kernels
       * on the device until we are sure that the data
       * is finished transfering)
       */
      cl::Event transfer_event;
      queue.enqueueWriteBuffer(oclBuffer_, CL_FALSE, 0,
                               sizeof(double) * rows * cols, A, NULL,
                               &transfer_event);
      this->add_write_event(transfer_event);
    } catch (const cl::Error& e) {
      check_opencl_error("matrix constructor", e);
    }
  }
  
  matrix_cl& operator=(const matrix_cl& a) {
    check_size_match("assignment of (OpenCL) matrices", "source.rows()",
                     a.rows(), "destination.rows()", rows());
    check_size_match("assignment of (OpenCL) matrices", "source.cols()",
                     a.cols(), "destination.cols()", cols());
    // Need to wait for all of matrices events before destroying old buffer
    this->wait_for_read_write_events();
    oclBuffer_ = a.buffer();
    return *this;
  }
};

}  // namespace math
}  // namespace stan

#endif
#endif
