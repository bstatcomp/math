#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_UNARY_FUNCTION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_UNARY_FUNCTION_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/err/check_matching_dims.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/opencl/kernel_generator/as_operation.hpp>
#include <stan/math/opencl/kernel_generator/is_usable_as_operation.hpp>
#include <string>
#include <type_traits>
#include <set>
#include <utility>

namespace stan {
namespace math {

template<typename Derived, typename T>
class unary_function : public operation<Derived, typename std::remove_reference_t<T>::ReturnScalar> {
public:
    using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
    static_assert(std::is_floating_point<ReturnScalar>::value, "unary_function: argument must be expression with floating point return type!");
    using base = operation<Derived, ReturnScalar>;
    using base::var_name;
    using base::instance;
    unary_function(T&& a, const std::string& fun) : a_(std::forward<T>(a)), fun_(fun) {}

    kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const{
      if(generated.count(instance)==0) {
        kernel_parts a_parts = a_.generate(ng, generated, i, j);
        generated.insert(instance);
        var_name = ng.generate();
        kernel_parts res;
        res.body = a_parts.body + type_str<ReturnScalar>::name + " " + var_name + " = " + fun_ + "(" + a_.var_name + ");\n";
        res.args = a_parts.args;
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
      }
    }

    void add_event(cl::Event& e) const{
      a_.add_event(e);
    }

    int rows() const{
      return a_.rows();
    }

    int cols() const{
      return a_.cols();
    }

    matrix_cl_view view() const{
      return a_.view();
    }

protected:
    const T a_;
    const std::string fun_;
};

#define ADD_UNARY_FUNCTION(fun) \
template<typename T> \
class fun##__ : public unary_function<fun##__<T>, T>{ \
public: \
    explicit fun##__(T&& a) : unary_function<fun##__<T>, T>(std::forward<T>(a),#fun){} \
    matrix_cl_view view() const{ \
      return matrix_cl_view::Entire; \
    } \
}; \
\
template<typename T, typename Cond = enable_if_none_arithmetic_all_usable_as_operation<T>> \
auto fun(T&& a) -> fun##__<decltype(as_operation(std::forward<T>(a)))>{ \
  return fun##__<decltype(as_operation(std::forward<T>(a)))>(as_operation(std::forward<T>(a))); \
}

#define ADD_UNARY_FUNCTION_PASS_ZERO(fun) \
template<typename T> \
class fun##__ : public unary_function<fun##__<T>, T>{ \
public: \
    explicit fun##__(T&& a) : unary_function<fun##__<T>, T>(std::forward<T>(a),#fun){} \
}; \
\
template<typename T, typename Cond = enable_if_none_arithmetic_all_usable_as_operation<T>> \
auto fun(T&& a) -> fun##__<decltype(as_operation(std::forward<T>(a)))>{ \
  return fun##__<decltype(as_operation(std::forward<T>(a)))>(as_operation(std::forward<T>(a))); \
}

ADD_UNARY_FUNCTION(rsqrt)
ADD_UNARY_FUNCTION_PASS_ZERO(sqrt)
ADD_UNARY_FUNCTION_PASS_ZERO(cbrt)

ADD_UNARY_FUNCTION(exp)
ADD_UNARY_FUNCTION(exp2)
ADD_UNARY_FUNCTION_PASS_ZERO(expm1)
ADD_UNARY_FUNCTION(log)
ADD_UNARY_FUNCTION(log2)
ADD_UNARY_FUNCTION(log10)
ADD_UNARY_FUNCTION_PASS_ZERO(log1p)

ADD_UNARY_FUNCTION_PASS_ZERO(sin)
ADD_UNARY_FUNCTION_PASS_ZERO(sinh)
ADD_UNARY_FUNCTION(cos)
ADD_UNARY_FUNCTION(cosh)
ADD_UNARY_FUNCTION_PASS_ZERO(tan)
ADD_UNARY_FUNCTION_PASS_ZERO(tanh)

ADD_UNARY_FUNCTION_PASS_ZERO(asin)
ADD_UNARY_FUNCTION_PASS_ZERO(asinh)
ADD_UNARY_FUNCTION(acos)
ADD_UNARY_FUNCTION(acosh)
ADD_UNARY_FUNCTION_PASS_ZERO(atan)
ADD_UNARY_FUNCTION_PASS_ZERO(atanh)

ADD_UNARY_FUNCTION(tgamma)
ADD_UNARY_FUNCTION(lgamma)
ADD_UNARY_FUNCTION_PASS_ZERO(erf)
ADD_UNARY_FUNCTION(erfc)

ADD_UNARY_FUNCTION_PASS_ZERO(floor)
ADD_UNARY_FUNCTION_PASS_ZERO(round)
ADD_UNARY_FUNCTION_PASS_ZERO(ceil)

#undef ADD_UNARY_FUNCTION
#undef ADD_UNARY_FUNCTION_PASS_ZERO

}
}
#endif
#endif
