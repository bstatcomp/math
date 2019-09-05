#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_BINARY_OPERATOR_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_BINARY_OPERATOR_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/err/check_matching_dims.hpp>
#include <stan/math/opencl/multiply.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/opencl/kernel_generator/scalar.hpp>
#include <stan/math/opencl/kernel_generator/as_operation.hpp>
#include <stan/math/opencl/kernel_generator/is_usable_as_operation.hpp>
#include <string>
#include <type_traits>
#include <set>
#include <utility>

namespace stan {
namespace math {

template<typename Derived, typename T_a, typename T_b>
class binary_operation : public operation<Derived, typename std::common_type<typename std::remove_reference_t<T_a>::ReturnScalar, typename std::remove_reference_t<T_b>::ReturnScalar>::type> {
public:
  static_assert(std::is_base_of<operation_base,std::remove_reference_t<T_a>>::value, "binary_operation: a must be an operation!");
  static_assert(std::is_base_of<operation_base,std::remove_reference_t<T_b>>::value, "binary_operation: b must be an operation!");

  using ReturnScalar = typename std::common_type<typename std::remove_reference_t<T_a>::ReturnScalar, typename std::remove_reference_t<T_b>::ReturnScalar>::type;
  using base = operation<Derived, ReturnScalar>;
  using base::var_name;
  using base::instance;

  binary_operation(T_a&& a, T_b&& b, const std::string& op) : a_(std::forward<T_a>(a)), b_(std::forward<T_b>(b)), op_(op) {
      const std::string function = "binary_operator" + op;
      if(a.rows()!=base::dynamic && b.rows()!=base::dynamic) {
        check_size_match(function.c_str(), "Rows of ", "a", a.rows(), "rows of ", "b", b.rows());
      }
      if(a.cols()!=base::dynamic && b.cols()!=base::dynamic) {
        check_size_match(function.c_str(), "Columns of ", "a", a.cols(), "columns of ", "b", b.cols());
      }
  }

  kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const{
    if(generated.count(instance)==0) {
      kernel_parts a_parts = a_.generate(ng, generated, i, j);
      kernel_parts b_parts = b_.generate(ng, generated, i, j);
      generated.insert(instance);
      var_name = ng.generate();
      kernel_parts res;
      res.body = a_parts.body + b_parts.body + type_str<ReturnScalar>::name + " " + var_name + " = " + a_.var_name + op_ + b_.var_name + ";\n";
      res.args = a_parts.args + b_parts.args;
      return res;
    }
    else{
      return {};
    }
  }

  void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const{
    if(generated.count(instance)==0) {
      generated.insert(instance);
      a_.set_args(generated, kernel, arg_num);
      b_.set_args(generated, kernel, arg_num);
    }
  }

  void add_event(cl::Event& e) const{
    a_.add_event(e);
    b_.add_event(e);
  }

  int rows() const{
    int a_rows = a_.rows();
    return a_rows == base::dynamic ? b_.rows() : a_rows;
  }

  int cols() const{
    int a_cols = a_.cols();
    return a_cols == base::dynamic ? b_.cols() : a_cols;
  }

  matrix_cl_view view() const{
    return either(a_.view(), b_.view());
  }

protected:
  T_a a_;
  T_b b_;
  std::string op_;
};


template<typename T_a, typename T_b>
class addition__ : public binary_operation<addition__<T_a, T_b>, T_a, T_b> {
public:
  addition__(T_a&& a, T_b&& b) : binary_operation<addition__<T_a, T_b>, T_a, T_b>(std::forward<T_a>(a),std::forward<T_b>(b),"+") {}
};

template<typename T_a, typename T_b, typename = enable_if_all_usable_as_operation<T_a, T_b>>
addition__<as_operation_t<T_a>,as_operation_t<T_b>> operator+(T_a&& a, T_b&& b) {
  return {as_operation(std::forward<T_a>(a)), as_operation(std::forward<T_b>(b))};
}


template<typename T_a, typename T_b>
class subtraction__ : public binary_operation<subtraction__<T_a, T_b>, T_a, T_b> {
public:
  subtraction__(T_a&& a, T_b&& b) : binary_operation<subtraction__<T_a, T_b>, T_a, T_b>(std::forward<T_a>(a),std::forward<T_b>(b),"-") {}
};

template<typename T_a, typename T_b, typename = enable_if_all_usable_as_operation<T_a, T_b>>
subtraction__<as_operation_t<T_a>,as_operation_t<T_b>> operator-(T_a&& a, T_b&& b){
  return {as_operation(std::forward<T_a>(a)), as_operation(std::forward<T_b>(b))};
}


template<typename T_a, typename T_b, typename = enable_if_all_usable_as_operation<T_a, T_b>>
class elewise_multiplication__ : public binary_operation<elewise_multiplication__<T_a, T_b>, T_a, T_b> {
public:
  elewise_multiplication__(T_a&& a, T_b&& b) : binary_operation<elewise_multiplication__<T_a, T_b>, T_a, T_b>(std::forward<T_a>(a),std::forward<T_b>(b),"*") {}

  matrix_cl_view view()  const{
    using base = binary_operation<elewise_multiplication__<T_a, T_b>, T_a, T_b>;
    return both(base::a_.view(), base::b_.view());
  }
};

template<typename T_a, typename T_b>
elewise_multiplication__<as_operation_t<T_a>,as_operation_t<T_b>> elewise_multiplication(T_a&& a, T_b&& b) {
  return {as_operation(std::forward<T_a>(a)), as_operation(std::forward<T_b>(b))};
}

template<typename T_a, typename T_b, typename = enable_if_arithmetic<T_a>, typename = enable_if_all_usable_as_operation<T_b>>
elewise_multiplication__<scalar__<T_a>,as_operation_t<T_b>> operator*(T_a&& a, T_b&& b) {
  return {as_operation(std::forward<T_a>(a)), as_operation(std::forward<T_b>(b))};
}

template<typename T_a, typename T_b, typename = enable_if_all_usable_as_operation<T_a>, typename = enable_if_arithmetic<T_b>>
elewise_multiplication__<as_operation_t<T_a>,scalar__<T_b>> operator*(T_a&& a, const T_b b) {
  return {as_operation(std::forward<T_a>(a)), as_operation(b)};
}

template<typename T_a, typename T_b, typename = enable_if_none_arithmetic_all_usable_as_operation<T_a, T_b>>
matrix_cl<double> operator*(const T_a& a, const T_b& b){
  //no need for perfect forwarding as operations are evaluated
  return as_operation(a).eval() * as_operation(b).eval();
}

template<typename T_a, typename T_b>
class elewise_division__ : public binary_operation<elewise_division__<T_a, T_b>, T_a, T_b> {
public:
  elewise_division__(T_a&& a, T_b&& b) : binary_operation<elewise_division__<T_a, T_b>, T_a, T_b>(std::forward<T_a>(a),std::forward<T_b>(b),"/") {}

  matrix_cl_view view()  const{
    using base = binary_operation<elewise_division__<T_a, T_b>, T_a, T_b>;
    return either(base::a_.view(), invert(base::b_.view()));
  }
};

template<typename T_a, typename T_b, typename = enable_if_all_usable_as_operation<T_a, T_b>>
elewise_division__<as_operation_t<T_a>,as_operation_t<T_b>> elewise_division(T_a&& a, T_b&& b) {
  return {as_operation(std::forward<T_a>(a)), as_operation(std::forward<T_b>(b))};
}

}
}
#endif
#endif
