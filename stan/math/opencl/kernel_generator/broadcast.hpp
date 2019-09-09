#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_BROADCAST_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_BROADCAST_HPP
#ifdef STAN_OPENCL

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

namespace stan {
namespace math {

/**
 * Represents a broadcasting operation in kernel generator expressions.
 * @tparam T type of argument
 * @tparam Rows whether to broadcast rows
 * @tparam Cols whether to broadcast columns
 */
template<typename T, bool Rows, bool Cols>
class broadcast__ : public operation<broadcast__<T, Rows, Cols>, typename std::remove_reference_t<T>::ReturnScalar> {
public:
  using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
  using base = operation<broadcast__<T, Rows, Cols>, ReturnScalar>;
  using base::var_name;
  using base::instance;

  /**
 * Constructor
 * @param a expression
 */
  explicit broadcast__(T&& a) : a_(std::forward<T>(a)) {
    const char* function = "broadcast";
    if (Rows) {
      check_size_match(function, "Rows of ", "a", a.rows(), "", "", 1);
    }
    if (Cols) {
      check_size_match(function, "Columns of ", "a", a.cols(), "", "", 1);
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
  inline kernel_parts generate(std::set<int>& generated, name_generator& ng, const std::string& i, const std::string& j) const {
    kernel_parts res = a_.generate(generated, ng, Rows ? "0" : i, Cols ? "0" : j);
    var_name = a_.var_name;
    return res;
  }

  /**
 * Sets kernel arguments for this and nested expressions.
 * @param[in,out] generated set of expressions that already set their kernel arguments
 * @param kernel kernel to set arguments on
 * @param[in,out] arg_num consecutive number of the first argument to set. This is incremented for each argument set by this function.
 */
  inline void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const {
    a_.set_args(generated, kernel, arg_num);
  }

  /**
 * Adds event for any matrices used by this or nested expressions.
 * @param e the event to add
 */
  inline void add_event(cl::Event& e) const {
    a_.add_event(e);
  }

  /**
 * Number of rows of a matrix that would be the result of evaluating this expression.
 * @return number of rows
 */
  inline int rows() const {
    return Rows ? base::dynamic : a_.rows();
  }

  /**
 * Number of columns of a matrix that would be the result of evaluating this expression.
 * @return number of columns
 */
  inline int cols() const {
    return Cols ? base::dynamic : a_.cols();
  }

  /**
 * View of a matrix that would be the result of evaluating this expression.
 * @return view
 */
  inline matrix_cl_view view() const {
    matrix_cl_view view = a_.view();
    if (Rows) {
      view = either(view, matrix_cl_view::Lower);
    }
    if (Cols) {
      view = either(view, matrix_cl_view::Upper);
    }
    return view;
  }

protected:
  T a_;
};

/**
 * Brodcast an expression in specified dimension(s). If broadcasting a dimension, that dimension of the input must be equal to 1.
 * Further expressions can use this expression as if had any size in broadcast dimension, repeating the values.
 * @tparam Rows whether to broadcast rows
 * @tparam Cols whether to broadcast rows
 * @tparam T type of input expression
 * @param a input expression
 * @return broadcast expression
 */
template<bool Rows, bool Cols, typename T, typename = enable_if_all_valid_expressions_and_none_scalar<T>>
inline broadcast__<as_operation_t<T>, Rows, Cols> broadcast(T&& a) {
  return broadcast__<as_operation_t<T>, Rows, Cols>(as_operation(std::forward<T>(a)));
}

}
}
#endif
#endif
