#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_CONDITIONAL_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_CONDITIONAL_HPP
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
 * Represents a ternary conditional operation in kernel generator expressions.
 * @tparam Derived derived type
 * @tparam T_condition type of condition
 * @tparam T_then type of then expression
 * @tparam T_else type of else expression
 */
template<typename T_condition, typename T_then, typename T_else>
class conditional__
        : public operation<conditional__<T_condition, T_then, T_else>, typename std::common_type<typename std::remove_reference_t<T_then>::ReturnScalar, typename std::remove_reference_t<T_else>::ReturnScalar>::type> {
public:
  using ReturnScalar = typename std::common_type<typename std::remove_reference_t<T_then>::ReturnScalar, typename std::remove_reference_t<T_else>::ReturnScalar>::type;
  using base = operation<conditional__<T_condition, T_then, T_else>, ReturnScalar>;
  using base::var_name;
  using base::instance;

  /**
   * Constructor
   * @param condition condition expression
   * @param then then expression
   * @param els else expression
   */
  conditional__(T_condition&& condition, T_then&& then, T_else&& els) :
          condition_(std::forward<T_condition>(condition)), then_(std::forward<T_then>(then)), else_(std::forward<T_else>(els)) {
    if (condition.rows() != base::dynamic && then.rows() != base::dynamic) {
      check_size_match("conditional", "Rows of ", "condition", condition.rows(), "rows of ", "then", then.rows());
    }
    if (condition.cols() != base::dynamic && then.cols() != base::dynamic) {
      check_size_match("conditional", "Columns of ", "condition", condition.cols(), "columns of ", "then", then.cols());
    }

    if (condition.rows() != base::dynamic && els.rows() != base::dynamic) {
      check_size_match("conditional", "Rows of ", "condition", condition.rows(), "rows of ", "else", els.rows());
    }
    if (condition.cols() != base::dynamic && els.cols() != base::dynamic) {
      check_size_match("conditional", "Columns of ", "condition", condition.cols(), "columns of ", "else", els.cols());
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
    if (generated.count(instance) == 0) {
      kernel_parts condition_parts = condition_.generate(generated, ng, i, j);
      kernel_parts then_parts = then_.generate(generated, ng, i, j);
      kernel_parts else_parts = else_.generate(generated, ng, i, j);
      generated.insert(instance);
      var_name = ng.generate();
      kernel_parts res;
      res.body =
              condition_parts.body + then_parts.body + else_parts.body + type_str<ReturnScalar>::name + " " + var_name + " = " + condition_.var_name + " ? " + then_.var_name + " : " + else_.var_name +
              ";\n";
      res.args = condition_parts.args + then_parts.args + else_parts.args;
      return res;
    }
    else {
      return {};
    }
  }

  /**
   * Sets kernel arguments for this and nested expressions.
   * @param[in,out] generated set of expressions that already set their kernel arguments
   * @param kernel kernel to set arguments on
   * @param[in,out] arg_num consecutive number of the first argument to set. This is incremented for each argument set by this function.
   */
  inline void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const {
    condition_.set_args(generated, kernel, arg_num);
    then_.set_args(generated, kernel, arg_num);
    else_.set_args(generated, kernel, arg_num);
  }

  /**
   * Adds event for any matrices used by this or nested expressions.
   * @param e the event to add
   */
  inline void add_event(cl::Event& e) const {
    condition_.add_event(e);
    then_.add_event(e);
    else_.add_event(e);
  }

  /**
  * Number of rows of a matrix that would be the result of evaluating this expression.
  * @return number of rows
  */
  inline int rows() const {
    return condition_.rows();
  }

  /**
   * Number of columns of a matrix that would be the result of evaluating this expression.
   * @return number of columns
   */
  inline int cols() const {
    return condition_.cols();
  }

  /**
   * View of a matrix that would be the result of evaluating this expression.
   * @return view
   */
  inline matrix_cl_view view() const {
    matrix_cl_view condition_view = condition_.view();
    matrix_cl_view then_view = then_.view();
    matrix_cl_view else_view = else_.view();
    return both(either(then_view, else_view), both(condition_view, then_view));
  }

protected:
  T_condition condition_;
  T_then then_;
  T_else else_;
};

/**
 * Ternary conditional operation on kernel generator expressions.
 * @tparam T_condition type of condition expression
 * @tparam T_then type of then expression
 * @tparam T_else type of else expression
 * @param condition condition expression
 * @param then then expression
 * @param els else expression
 * @return conditional operation expression
 */
template<typename T_condition, typename T_then, typename T_else, typename = enable_if_all_valid_expressions<T_condition, T_then, T_else>>
inline conditional__<as_operation_t<T_condition>, as_operation_t<T_then>, as_operation_t<T_else>> conditional(T_condition&& condition, T_then&& then, T_else&& els) {
  return {as_operation(std::forward<T_condition>(condition)),
          as_operation(std::forward<T_then>(then)),
          as_operation(std::forward<T_else>(els))};
}

}
}
#endif
#endif
