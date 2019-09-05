#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_REDUCTION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_REDUCTION_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl_view.hpp>
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

template<typename Derived, typename T, typename Operation, bool PassZero, bool Rowwise, bool Colwise>
class reduction : public operation<Derived, typename std::remove_reference_t<T>::ReturnScalar> {
public:
    using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
    using base = operation<Derived, ReturnScalar>;
    using base::var_name;
    using base::instance;

    reduction(T&& a, const std::string& init) : a_(std::forward<T>(a)), init_(init) {}

    kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const {
      if (generated.count(instance) == 0) {
        generated.insert(instance);
        var_name = ng.generate();
        std::set<int> generated_internal;
        kernel_parts a_parts = a_.generate(ng, generated_internal, Colwise ? var_name + "_i" : i, Rowwise ? var_name + "_j" : j);
        kernel_parts res;
        res.body = type_str<ReturnScalar>::name + " " + var_name + " = " + init_ + ";\n";
        if (Rowwise) {
          if(PassZero) {
            res.body += "for(int " + var_name + "_j = contains_nonzero(" + var_name + "_view, LOWER) ? 0 : " + i + "; "
                        + var_name + "_j < (contains_nonzero(" + var_name + "_view, UPPER) ? " + var_name + "_cols : " + i + " + 1); " + var_name + "_j++){\n";
          }
          else{
            res.body += "for(int " + var_name + "_j = 0; " + var_name + "_j < " + var_name + "_cols; " + var_name + "_j++){\n";
          }
        }
        if (Colwise) {
          if(PassZero) {
            res.body += "for(int " + var_name + "_i = contains_nonzero(" + var_name + "_view, UPPER) ? 0 : " + j + "; "
                    + var_name + "_i < (contains_nonzero(" + var_name + "_view, LOWER) ? " + var_name + "_rows : " + j + " + 1); " + var_name + "_i++){\n";
          }
          else{
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

    void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const {
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

    void add_event(cl::Event& e) const {
      a_.add_event(e);
    }


    int rows() const {
      if (Colwise) {
        return 1;
      }
      else {
        return a_.rows();
      }
    }

    int cols() const {
      if (Rowwise) {
        return 1;
      }
      else {
        return a_.cols();
      }
    }

    matrix_cl_view view() const {
      return a_.view();
    }

protected:
    T a_;
    std::string init_;
};

//template<typename Derived, typename T, typename Operation, bool PassZero>
//class reduction<Derived, T, Operation, PassZero, true, true> : public operation<Derived, typename std::remove_reference_t<T>::ReturnScalar> {
//public:
//    using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
//    using base = operation<Derived, ReturnScalar>;
//    using base::var_name;
//    using base::instance;
//
//    reduction(T&& a, const std::string& init) : a_(std::forward<T>(a)), init_(init) {}
//
//};

struct sum_op {
    static std::string generate(const std::string& a, const std::string& b) {
      return a + " + " + b;
    }
};

template<typename T, bool Rowwise, bool Colwise>
class sum__ : public reduction<sum__<T, Rowwise, Colwise>, T, sum_op, true, Rowwise, Colwise> {
public:
    explicit sum__(T&& a) : reduction<sum__<T, Rowwise, Colwise>, T, sum_op, true, Rowwise, Colwise>(std::forward<T>(a), "0") {}
};

template<bool Rowwise, bool Colwise, typename T, typename = enable_if_none_arithmetic_all_usable_as_operation <T>>
auto sum(T&& a) -> const sum__<decltype(as_operation(std::forward<T>(a))), Rowwise, Colwise> {
  return sum__<decltype(as_operation(std::forward<T>(a))), Rowwise, Colwise>(as_operation(std::forward<T>(a)));
}

template< typename T>
struct max_op {
    static std::string generate(const std::string& a, const std::string& b) {
      if(std::is_floating_point<T>()){
        return "fmax(" + a + ", " + b + ")";
      }
      return "max(" + a + ", " + b + ")";
    }
};

template<typename T, bool Rowwise, bool Colwise>
class max__ : public reduction<max__<T, Rowwise, Colwise>, T, max_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise> {
public:
    explicit max__(T&& a) : reduction<max__<T, Rowwise, Colwise>, T, max_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise>(std::forward<T>(a), "-INFINITY") {}
};

template<bool Rowwise, bool Colwise, typename T, typename = enable_if_none_arithmetic_all_usable_as_operation <T>>
auto max(T&& a) -> const max__<decltype(as_operation(std::forward<T>(a))), Rowwise, Colwise> {
  return max__<decltype(as_operation(std::forward<T>(a))), Rowwise, Colwise>(as_operation(std::forward<T>(a)));
}


template< typename T>
struct min_op {
    static std::string generate(const std::string& a, const std::string& b) {
      if(std::is_floating_point<T>()){
        return "fmin(" + a + ", " + b + ")";
      }
      return "min(" + a + ", " + b + ")";
    }
};

template<typename T, bool Rowwise, bool Colwise>
class min__ : public reduction<min__<T, Rowwise, Colwise>, T, min_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise> {
public:
    explicit min__(T&& a) : reduction<min__<T, Rowwise, Colwise>, T, min_op<typename std::remove_reference_t<T>::ReturnScalar>, false, Rowwise, Colwise>(std::forward<T>(a), "INFINITY") {}
};

template<bool Rowwise, bool Colwise, typename T, typename = enable_if_none_arithmetic_all_usable_as_operation <T>>
auto min(T&& a) -> const min__<decltype(as_operation(std::forward<T>(a))), Rowwise, Colwise> {
  return min__<decltype(as_operation(std::forward<T>(a))), Rowwise, Colwise>(as_operation(std::forward<T>(a)));
}

}
}

#endif
#endif
