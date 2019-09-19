#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_BLOCK_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_BLOCK_HPP
#ifdef  STAN_OPENCL

#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/opencl/kernel_generator/as_operation.hpp>
#include <stan/math/opencl/kernel_generator/is_valid_expression.hpp>
#include <string>
#include <type_traits>
#include <set>
#include <utility>

namespace stan{
namespace math{

/**
 * Represents submatrix block in kernel generator expressions.
 * @tparam Derived derived type
 * @tparam T type of argument
 */
template<typename T>
class block__ : public operation<block__<T>, typename std::remove_reference_t<T>::ReturnScalar> {
public:
  using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
  using base = operation<block__<T>, ReturnScalar>;
  using base::var_name;
  using base::instance;

  /**
 * Constructor
 * @param a expression
 * @param start_row first row of block
 * @param start_col first column of a block
 * @param rows number of rows in block
 * @param cols number of columns in block
 */
  block__(T&& a, int start_row, int start_col, int rows, int cols) :
  a_(std::forward<T>(a)), start_row_(start_row), start_col_(start_col), rows_(rows), cols_(cols) {
    if ((a.rows() != base::dynamic && (start_row + rows) > a.rows()) ||
        (a.cols() != base::dynamic && (start_col + cols) > a.cols())) {
      domain_error("block", "block of \"a\"", " is out of bounds", "");
    }
  }

  /**
   * generates kernel code for this and nested expressions.
 * @param[in,out] generated set of already generated operations
 * @param ng name generator for this kernel
 * @param i row index variable name
   * @param j column index variable name
   * @return part of kernel with code for this and nested expressions
   */
  inline kernel_parts generate(std::set<int>& generated, name_generator& ng, const std::string& i, const std::string& j) const{
    kernel_parts res = a_.generate(generated, ng, "(" + i + " + " + std::to_string(start_row_) + ")", "(" + j + " + " + std::to_string(start_col_) + ")");
    var_name = a_.var_name;
    return res;
  }

  /**
   * generates kernel code for this and nested expressions if this expression appears on the left hand side of an assignment.
 * @param[in,out] generated set of already generated operations
 * @param ng name generator for this kernel
 * @param i row index variable name
   * @param j column index variable name
   * @return part of kernel with code for this and nested expressions
   */
  inline kernel_parts generate_lhs(std::set<int>& generated, name_generator& ng, const std::string& i, const std::string& j) const{
    return a_.generate_lhs(generated, ng, "(" + i + " + " + std::to_string(start_row_) + ")", "(" + j + " + " + std::to_string(start_col_) + ")");
  }

  /**
 * Sets kernel arguments for this and nested expressions.
 * @param[in,out] generated set of expressions that already set their kernel arguments
 * @param kernel kernel to set arguments on
 * @param[in,out] arg_num consecutive number of the first argument to set. This is incremented for each argument set by this function.
 */
  inline void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const{
    if(generated.count(instance)==0) {
      generated.insert(instance);
      a_.set_args(generated, kernel, arg_num);
    }
  }

  /**
 * Adds event for any matrices used by this or nested expressions.
 * @param e the event to add
 */
  inline void add_event(cl::Event& e) const {
    a_.add_event(e);
  }

  /**
 * Adds write event for any matrices used by this or nested expressions.
 * @param e the event to add
 */
  inline void add_write_event(cl::Event& e) const {
    a_.add_event(e);
  }

  /**
   * Number of rows of a matrix that would be the result of evaluating this expression.
   * @return number of rows
   */
  inline int rows() const {
    return rows_;
  }

  /**
 * Number of columns of a matrix that would be the result of evaluating this expression.
 * @return number of columns
 */
  inline int cols() const {
    return cols_;
  }

    /**
   * View of a matrix that would be the result of evaluating this expression.
   * @return view
   */
  inline matrix_cl_view view() const {
    return transpose(a_.view());
  }

  /**
   * Evaluates an expression and assigns it to the block.
   * @tparam T_expression type of expression
   * @param rhs input expression
   */
  template<typename T_expression, typename = enable_if_all_valid_expressions_and_none_scalar<T>>
  const block__<T>& operator= (T_expression&& rhs) const{
    check_size_match("block.operator=", "Rows of ", "rhs", rhs.rows(), "rows of ", "*this", this->rows());
    check_size_match("block.operator=", "Cols of ", "rhs", rhs.cols(), "cols of ", "*this", this->cols());
    auto expression = as_operation(std::forward<T_expression>(rhs));
    expression.evaluate_into(*this);
    return *this;
  }

protected:
  T a_;
  int start_row_, start_col_, rows_, cols_;
};

/**
 * Block of a kernel generator expression.
 * @tparam T type of argument
 * @param a input argument
 * @return Block of given expression
 */
template<typename T, typename = enable_if_all_valid_expressions_and_none_scalar<T>>
inline block__<as_operation_t<T>> block(T&& a, int start_row, int start_col, int rows, int cols) {
  return block__<as_operation_t<T>>(as_operation(std::forward<T>(a)), start_row, start_col, rows, cols);
}

}
}

#endif
#endif
