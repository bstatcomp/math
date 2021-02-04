#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_RESHAPE_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_RESHAPE_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/kernel_generator/operation_cl.hpp>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace stan {
namespace math {

/** \addtogroup opencl_kernel_generator
 *  @{
 */

/**
 * Represents a matrix of single repeated value in kernel generator expressions.
 * @tparam T type of the scalar
 */
template <typename T>
class reshape_ : public operation_cl<reshape_<T>, scalar_type_t<T>, T> {
  int rows_;
  int cols_;

 public:
  using Scalar = scalar_type_t<T>;
  using base = operation_cl<reshape_<T>, Scalar, T>;
  using base::var_name_;

  /**
   * Constructor
   * @param a scalar value
   * @param rows number of rows of the matrix
   * @param cols number of columns of the matrix
   */
  explicit reshape_(T&& a, int rows, int cols)
      : base(std::forward<T>(a)), rows_(rows), cols_(cols) {
    check_size_match("reshape", "argument size", a.size(), "new size",
                     rows * cols);
  }

  /**
   * Creates a deep copy of this expression.
   * @return copy of \c *this
   */
  inline reshape_<T> deep_copy() const {
    auto&& arg_copy = this->template get_arg<0>().deep_copy();
    return reshape_<std::remove_reference_t<decltype(arg_copy)>>{
        std::move(arg_copy), rows_, cols_};
  }

  /**
   * Generates kernel code for this expression.
   * @param row_index_name row index variable name
   * @param col_index_name column index variable name
   * @param view_handled whether whether caller already handled matrix view
   * @return part of kernel with code for this expression
   */
  inline kernel_parts generate(const std::string& row_index_name,
                               const std::string& col_index_name,
                               const bool view_handled,
                               const std::string& var_name_arg) const {
    kernel_parts res{};
    res.args = "int " + var_name_ + "_inner_rows, int " + var_name_
               + "_outer_rows, ";
    res.body_prefix
        = "int " + var_name_ + "_lin = " + var_name_ + "_outer_rows * "
          + col_index_name + " + " + row_index_name + ";\n"
          "int " + var_name_ + "_col_idx = " + var_name_ + "_lin / "
          + var_name_ + "_inner_rows;\n"
          "int " + var_name_ + "_row_idx = " + var_name_ + "_lin % "
          + var_name_ + "_inner_rows;\n";
    res.body
        = type_str<Scalar>() + " " + var_name_ + " = " + var_name_arg + ";\n";
    return res;
  }
  /**
   * Sets indices for the argument expression.
   * @param[in, out] row_index_name row index
   * @param[in, out] col_index_name column index
   */
  inline void modify_argument_indices(std::string& row_index_name,
                                      std::string& col_index_name) const {
    row_index_name = var_name_ + "_row_idx";
    col_index_name = var_name_ + "_col_idx";
  }

  /**
   * Sets kernel arguments for this and nested expressions.
   * @param[in,out] generated map from (pointer to) already generated local
   * operations to variable names
   * @param[in,out] generated_all map from (pointer to) already generated all
   * operations to variable names
   * @param kernel kernel to set arguments on
   * @param[in,out] arg_num consecutive number of the first argument to set.
   * This is incremented for each argument set by this function.
   */
  inline void set_args(std::map<const void*, const char*>& generated,
                       std::map<const void*, const char*>& generated_all,
                       cl::Kernel& kernel, int& arg_num) const {
    if (generated.count(this) == 0) {
      generated[this] = "";
      std::map<const void*, const char*> generated2;
      this->template get_arg<0>().set_args(generated2, generated_all, kernel,
                                           arg_num);
      if (generated_all.count(this) == 0) {
        generated_all[this] = "";
        kernel.setArg(arg_num++, this->template get_arg<0>().rows());
        kernel.setArg(arg_num++, rows_);
      }
    }
  }

  /**
   * Number of rows of a matrix that would be the result of evaluating this
   * expression.
   * @return number of rows
   */
  inline int rows() const { return rows_; }

  /**
   * Number of columns of a matrix that would be the result of evaluating this
   * expression.
   * @return number of columns
   */
  inline int cols() const { return cols_; }

  /**
   * Determine indices of extreme sub- and superdiagonals written.
   * @return pair of indices - bottom and top diagonal
   */
  inline std::pair<int, int> extreme_diagonals() const {
    return {std::numeric_limits<int>::min(), std::numeric_limits<int>::max()};
  }
};

/**
 * Matrix of repeated values in kernel generator expressions.
 *
 * In most cases scalars should be directly used instead of this. This is,
 * however, useful for initializing some expression to specific value if that
 * expresssion could also be plain `matrix_cl`.
 *
 * @tparam T type of argument
 * @param a input argument
 * @param rows number of rows
 * @param cols number of columns
 * @return Block of given expression
 */
template <typename T,
          require_all_kernel_expressions_and_none_scalar_t<T>* = nullptr>
inline auto reshape(T&& a, int rows, int cols) {
  auto&& a_operation = as_operation_cl(std::forward<T>(a)).deep_copy();
  return reshape_<std::remove_reference_t<decltype(a_operation)>>(
      std::move(a_operation), rows, cols);
}

/** @}*/
}  // namespace math
}  // namespace stan

#endif
#endif
