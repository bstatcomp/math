#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_REDUCTION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_REDUCTION_HPP
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
 * Represents a reduction in kernel generator expressions.
 * @tparam Derived derived type
 * @tparam T type of first argument
 * @tparam Operation type with member function generate that accepts two variable names and returns OpenCL source code for reduction operation
 * @tparam PassZero whether \c Operation passes trough zeros
 * @tparam Rowwise whether this is row wise reduction
 * @tparam Colwise whether this is column wise reduction
 */

template<typename Derived, typename T, typename Operation, bool PassZero, bool Rowwise, bool Colwise>
class reduction : public operation<Derived, typename std::remove_reference_t<T>::ReturnScalar> {
public:
  using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
  using base = operation<Derived, ReturnScalar>;
  using base::var_name;
  using base::instance;

  /**
   * Constructor
   * @param a the expression to reduce
   * @param init OpenCL source code of initialization value for reduction
   */
  reduction(T&& a, const std::string& init) : a_(std::forward<T>(a)), init_(init) {}

  /**
   * generates kernel code for this and nested expressions.
    * @param[in,out] generated set of already generated operations* @param ng name generator for this kernel
   * @param i row index variable name
   * @param j column index variable name
   * @return part of kernel with code for this and nested expressions
   */
  inline kernel_parts generate(std::set<int>& generated, name_generator& ng, const std::string& i, const std::string& j) const {
    if (generated.count(instance) == 0) {
      generated.insert(instance);
      var_name = ng.generate();
      std::set<int> generated_internal;
      kernel_parts a_parts = a_.generate(generated_internal, ng, Colwise ? var_name + "_i" : i, Rowwise ? var_name + "_j" : j);
      kernel_parts res;
      res.body = type_str<ReturnScalar>::name + " " + var_name + " = " + init_ + ";\n";
      if (Rowwise) {
        if (PassZero) {
          res.body += "for(int " + var_name + "_j = contains_nonzero(" + var_name + "_view, LOWER) ? 0 : " + i + "; "
                      + var_name + "_j < (contains_nonzero(" + var_name + "_view, UPPER) ? " + var_name + "_cols : " + i + " + 1); " + var_name + "_j++){\n";
        }
        else {
          res.body += "for(int " + var_name + "_j = 0; " + var_name + "_j < " + var_name + "_cols; " + var_name + "_j++){\n";
        }
      }
      if (Colwise) {
        if (PassZero) {
          res.body += "for(int " + var_name + "_i = contains_nonzero(" + var_name + "_view, UPPER) ? 0 : " + j + "; "
                      + var_name + "_i < (contains_nonzero(" + var_name + "_view, LOWER) ? " + var_name + "_rows : " + j + " + 1); " + var_name + "_i++){\n";
        }
        else {
          res.body += "for(int " + var_name + "_i = 0; " + var_name + "_i < " + var_name + "_rows; " + var_name + "_i++){\n";
        }
      }
      res.body += a_parts.body + var_name + " = " + Operation::generate(var_name, a_.var_name) + ";\n";
      if (Rowwise) {
        res.body += "}\n";
      }
      if (Colwise) {
        res.body += "}\n";
      }
      res.args = a_parts.args + "int " + var_name + "_view, ";
      if (Rowwise) {
        res.args += "int " + var_name + "_cols, ";
      }
      if (Colwise) {
        res.args += "int " + var_name + "_rows, ";
      }
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
    if (generated.count(instance) == 0) {
      generated.insert(instance);
      a_.set_args(generated, kernel, arg_num);
      kernel.setArg(arg_num++, a_.view());
      if (Rowwise) {
        kernel.setArg(arg_num++, a_.cols());
      }
      if (Colwise) {
        kernel.setArg(arg_num++, a_.rows());
      }
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
    if (Colwise) {
      return 1;
    }
    else {
      return a_.rows();
    }
  }

  /**
   * Number of columns of a matrix that would be the result of evaluating this expression.
   * @return number of columns
   */
  inline int cols() const {
    if (Rowwise) {
      return 1;
    }
    else {
      return a_.cols();
    }
  }

  /**
   * View of a matrix that would be the result of evaluating this expression.
   * @return view
   */
  matrix_cl_view view() const {
    return matrix_cl_view::Entire;
  }

protected:
  T a_;
  std::string init_;
};

/**
 * operation for sum reduction.
 */
struct sum_op {
  /**
   * Generates sum reduction
   * @param a first variable
   * @param b second variable
   * @return reduction code
   */
  inline static std::string generate(const std::string& a, const std::string& b) {
    return a + " + " + b;
  }
};

/**
 * Represents sum - reduction in kernel generator expressions.
 * @tparam T type of expression
 * @tparam Rowwise whether to sum row wise
 * @tparam Colwise whether to sum column wise
 */
template<typename T, bool Rowwise, bool Colwise>
class sum__ : public reduction<sum__<T, Rowwise, Colwise>, T, sum_op, true, Rowwise, Colwise> {
public:
  explicit sum__(T&& a) : reduction<sum__<T, Rowwise, Colwise>, T, sum_op, true, Rowwise, Colwise>(std::forward<T>(a), "0") {}
};

/**
 * Sum - reduction of a kernel generator expression.
 * @tparam Rowwise whether to sum row wise
 * @tparam Colwise whether to sum column wise
 * @tparam T type of input expression
 * @param a expression to reduce
 * @return sum
 */
template<bool Rowwise, bool Colwise, typename T, typename = enable_if_all_valid_expressions_and_none_scalar<T>>
inline sum__<as_operation_t<T>, Rowwise, Colwise> sum(T&& a) {
  return sum__<as_operation_t<T>, Rowwise, Colwise>(as_operation(std::forward<T>(a)));
}

/**
 * operation for max reduction.
 * @tparam T type to reduce
 */
template<typename T>
struct max_op {
  /**
   * Generates sum reduction
   * @param a first variable
   * @param b second variable
   * @return reduction code
   */
  inline static std::string generate(const std::string& a, const std::string& b) {
    if (std::is_floating_point<T>()) {
      return "fmax(" + a + ", " + b + ")";
    }
    return "max(" + a + ", " + b + ")";
  }
};

/**
 * Represents max - reduction in kernel generator expressions.
 * @tparam T type of expression
 * @tparam Rowwise whether to reduce row wise
 * @tparam Colwise whether to reduce column wise
 */
template<typename T, bool Rowwise, bool Colwise>
class max__ : public reduction<max__<T, Rowwise, Colwise>, T, max_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise> {
public:
  explicit max__(T&& a) : reduction<max__<T, Rowwise, Colwise>, T, max_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise>(std::forward<T>(a), "-INFINITY") {}
};

/**
 * Max - reduction of a kernel generator expression.
 * @tparam Rowwise whether to reduce row wise
 * @tparam Colwise whether to reduce column wise
 * @tparam T type of input expression
 * @param a expression to reduce
 * @return max
 */
template<bool Rowwise, bool Colwise, typename T, typename = enable_if_all_valid_expressions_and_none_scalar<T>>
inline max__<as_operation_t<T>, Rowwise, Colwise> max(T&& a) {
  return max__<as_operation_t<T>, Rowwise, Colwise>(as_operation(std::forward<T>(a)));
}

/**
 * operation for min reduction.
 * @tparam T type to reduce
 */
template<typename T>
struct min_op {
  /**
   * Generates sum reduction
   * @param a first variable
   * @param b second variable
   * @return reduction code
   */
  inline static std::string generate(const std::string& a, const std::string& b) {
    if (std::is_floating_point<T>()) {
      return "fmin(" + a + ", " + b + ")";
    }
    return "min(" + a + ", " + b + ")";
  }
};

/**
 * Represents min - reduction in kernel generator expressions.
 * @tparam T type of expression
 * @tparam Rowwise whether to reduce row wise
 * @tparam Colwise whether to reduce column wise
 */
template<typename T, bool Rowwise, bool Colwise>
class min__ : public reduction<min__<T, Rowwise, Colwise>, T, min_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise> {
public:
  explicit min__(T&& a) : reduction<min__<T, Rowwise, Colwise>, T, min_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise>(std::forward<T>(a), "INFINITY") {}
};

/**
 * Min - reduction of a kernel generator expression.
 * @tparam Rowwise whether to reduce row wise
 * @tparam Colwise whether to reduce column wise
 * @tparam T type of input expression
 * @param a expression to reduce
 * @return min
 */
template<bool Rowwise, bool Colwise, typename T, typename = enable_if_all_valid_expressions_and_none_scalar<T>>
inline min__<as_operation_t<T>, Rowwise, Colwise> min(T&& a) {
  return min__<as_operation_t<T>, Rowwise, Colwise>(as_operation(std::forward<T>(a)));
}

}
}

#endif
#endif
