#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_TRANSPOSE_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_TRANSPOSE_HPP
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

namespace stan {
namespace math {

/**
 * Represents a transpose in kernel generator expressions.
 * @tparam Derived derived type
 * @tparam T_a type of first argument
 * @tparam T_b type of second argument
 */
template<typename T>
class transpose__ : public operation<transpose__<T>, typename std::remove_reference_t<T>::ReturnScalar> {
public:
  using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
  using base = operation<transpose__<T>, ReturnScalar>;
  using base::var_name;
  using base::instance;

  /**
   * Constructor
   * @param a expression to transpose
   */
  explicit transpose__(T&& a) : a_(std::forward<T>(a)) {}

  /**
 * generates kernel code for this and nested expressions.
 * @param[in,out] generated set of already generated operations
 * @param ng name generator for this kernel
 * @param i row index variable name
 * @param j column index variable name
 * @return part of kernel with code for this and nested expressions
 */
  inline kernel_parts generate(std::set<int>& generated, name_generator& ng, const std::string& i, const std::string& j) const {
    kernel_parts res = a_.generate(generated, ng, j, i);
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
    if (generated.count(instance) == 0) {
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
 * Number of rows of a matrix that would be the result of evaluating this expression.
 * @return number of rows
 */
  inline int rows() const {
    return a_.cols();
  }

  /**
 * Number of columns of a matrix that would be the result of evaluating this expression.
 * @return number of columns
 */
  inline int cols() const {
    return a_.rows();
  }

  /**
 * View of a matrix that would be the result of evaluating this expression.
 * @return view
 */
  inline matrix_cl_view view() const {
    return transpose(a_.view());
  }

protected:
  T a_;
};

template<typename T, typename = enable_if_all_valid_expressions_and_none_scalar<T>>
inline transpose__<as_operation_t<T>> transpose(T&& a) {
  return transpose__<as_operation_t<T>>(as_operation(std::forward<T>(a)));
}

}
}

#endif
#endif
