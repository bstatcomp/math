#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_BINARY_OPERATOR_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_BINARY_OPERATOR_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/err/check_matching_dims.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <string>

namespace stan {
namespace math {

template<typename T1, typename T2>
class binary_operation : public operation<return_type_t<T1, T2>> {
public:
  using Type = return_type_t<T1, T2>;

  binary_operation(const operation<T1>& a, const operation<T2>& b, std::string op) : a_(a), b_(b), op_(op) {
    check_matching_dims(("binary_operator" + op).c_str(), "a", a, "b", b);
  }

  virtual kernel_parts generate(const std::string& i, const std::string& j) const{
    kernel_parts a_parts = a_.generate(i, j);
    kernel_parts b_parts = b_.generate(i, j);
    kernel_parts res;
    res.body = a_parts.body + b_parts.body + type_str<Type>::name + " " + this->var_name + " = " + a_.var_name + op_ + b_.var_name + ";\n";
    res.args = a_parts.args + b_parts.args;
    return res;
  }

  virtual void set_args(cl::Kernel& kernel, int& arg_num) const{
    a_.set_args(kernel,arg_num);
    b_.set_args(kernel,arg_num);
  }

  virtual void add_event(cl::Event& e) const{
    a_.add_event(e);
    b_.add_event(e);
  }

  virtual int rows() const{
    return a_.rows();
  }

  virtual int cols() const{
    return a_.cols();
  }

  virtual matrix_cl_view view() const{
    return either(a_.view(), b_.view());
  }

protected:
  const operation<T1>& a_;
  const operation<T2>& b_;
  const std::string op_;
};


template<typename T1, typename T2>
class addition : public binary_operation<T1, T2> {
public:
  addition(const operation<T1>& a, const operation<T2>& b) : binary_operation<T1, T2>(a,b,"+") {}
};


template<typename T1, typename T2>
class subtraction : public binary_operation<T1, T2> {
public:
  subtraction(const operation<T1>& a, const operation<T2>& b) : binary_operation<T1, T2>(a,b,"-") {}
};


template<typename T1, typename T2>
class elewiseMultiplication : public binary_operation<T1, T2> {
public:
  elewiseMultiplication(const operation<T1>& a, const operation<T2>& b) : binary_operation<T1, T2>(a,b,"*") {}

//  virtual matrix_cl_view view()  const{
//    return both(a_.view(), b_.view());
//  }
};


//template<typename T1, typename T2>
//class elewiseDivision : public binary_operation<T1, T2> {
//public:
//  elewiseDivision(const operation<T1>& a, const operation<T2>& b) : binary_operation(a,b,"/") {}
//
////  virtual matrix_cl_view view()  const{
////    return either(a_.view(), inverse(b_.view()));
////  }
//};

}
}
#endif
#endif
