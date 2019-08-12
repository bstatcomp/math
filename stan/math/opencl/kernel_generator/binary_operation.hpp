#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_BINARY_OPERATOR_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_BINARY_OPERATOR_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/err/check_matching_dims.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/opencl/kernel_generator/as_operation.hpp>
#include <string>
#include <type_traits>

namespace stan {
namespace math {

template<typename Derived, typename T_a, typename T_b>
class binary_operation : public operation<Derived, return_type_t<T_a, T_b>> {
public:
  static_assert(std::is_base_of<operation_base,T_a>::value, "binary_operation: a must be an operation!");
  static_assert(std::is_base_of<operation_base,T_b>::value, "binary_operation: b must be an operation!");
  using T = return_type_t<T_a, T_b>;
  using base = operation<Derived, return_type_t<T_a, T_b>>;
//  using operation<Derived, T>::ReturnScalar;
  binary_operation(const T_a& a, const T_b& b, const std::string& op) : a_(a), b_(b), op_(op) {
      const std::string function = "binary_operator" + op;
      if(a.rows()!=base::dynamic && b.rows()!=base::dynamic) {
        check_size_match(function.c_str(), "Rows of ", "a", a.rows(), "rows of ", "b", b.rows());
      }
      if(a.cols()!=base::dynamic && b.cols()!=base::dynamic) {
        check_size_match(function.c_str(), "Columns of ", "a", a.cols(), "columns of ", "b", b.cols());
      }
  }

  kernel_parts generate(const std::string& i, const std::string& j) const{
    kernel_parts a_parts = a_.generate(i, j);
    kernel_parts b_parts = b_.generate(i, j);
    kernel_parts res;
    res.body = a_parts.body + b_parts.body + type_str<T>::name + " " + this->var_name + " = " + a_.var_name + op_ + b_.var_name + ";\n";
    res.args = a_parts.args + b_parts.args;
    return res;
  }

  void set_args(cl::Kernel& kernel, int& arg_num) const{
    a_.set_args(kernel,arg_num);
    b_.set_args(kernel,arg_num);
  }

  void add_event(cl::Event& e) const{
    a_.add_event(e);
    b_.add_event(e);
  }

  int rows() const{
    return a_.rows();
  }

  int cols() const{
    return a_.cols();
  }

  matrix_cl_view view() const{
    return either(a_.view(), b_.view());
  }

protected:
  const T_a a_;
  const T_b b_;
  const std::string op_;
};


template<typename T_a, typename T_b>
class addition__ : public binary_operation<addition__<T_a, T_b>, T_a, T_b> {
public:
  addition__(const T_a& a, const T_b& b) : binary_operation<addition__<T_a, T_b>, T_a, T_b>(a,b,"+") {}
};

template<typename T_a, typename T_b>
auto addition(const T_a& a, const T_b& b) -> const addition__<decltype(as_operation(a)),decltype(as_operation(b))>{
  return {as_operation(a), as_operation(b)};
}


template<typename T_a, typename T_b>
class subtraction__ : public binary_operation<subtraction__<T_a, T_b>, T_a, T_b> {
public:
  subtraction__(const T_a& a, const T_b& b) : binary_operation<subtraction__<T_a, T_b>, T_a, T_b>(a,b,"-") {}
};

template<typename T_a, typename T_b>
auto subtraction(const T_a& a, const T_b& b) -> const subtraction__<decltype(as_operation(a)),decltype(as_operation(b))>{
  return {as_operation(a), as_operation(b)};
}


template<typename T_a, typename T_b>
class elewise_multiplication__ : public binary_operation<elewise_multiplication__<T_a, T_b>, T_a, T_b> {
public:
  elewise_multiplication__(const T_a& a, const T_b& b) : binary_operation<elewise_multiplication__<T_a, T_b>, T_a, T_b>(a,b,"*") {}

  matrix_cl_view view()  const{
    using base = binary_operation<elewise_multiplication__<T_a, T_b>, T_a, T_b>;
    return both(base::a_.view(), base::b_.view());
  }
};

template<typename T_a, typename T_b>
auto elewise_multiplication(const T_a& a, const T_b& b) -> const elewise_multiplication__<decltype(as_operation(a)),decltype(as_operation(b))>{
  return {as_operation(a), as_operation(b)};
}


template<typename T_a, typename T_b>
class elewise_division__ : public binary_operation<elewise_division__<T_a, T_b>, T_a, T_b> {
public:
  using T = return_type_t<T_a, T_b>;
  elewise_division__(const T_a& a, const T_b& b) : binary_operation<elewise_division__<T_a, T_b>, T_a, T_b>(a,b,"/") {}

  matrix_cl_view view()  const{
    using base = binary_operation<elewise_division__<T_a, T_b>, T_a, T_b>;
    return either(base::a_.view(), invert(base::b_.view()));
  }
};

template<typename T_a, typename T_b>
auto elewise_division(const T_a& a, const T_b& b) -> const elewise_division__<decltype(as_operation(a)),decltype(as_operation(b))>{
  return {as_operation(a), as_operation(b)};
}

}
}
#endif
#endif
