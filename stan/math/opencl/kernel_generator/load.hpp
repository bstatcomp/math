#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_LOAD_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_LOAD_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <type_traits>
#include <string>
#include <utility>
#include <set>

namespace stan {
namespace math {

template<typename T>
class load__ : public operation<load__<T>, typename std::remove_reference_t<T>::type>{
public:
    using ReturnScalar = typename std::remove_reference_t<T>::type;
    using base = operation<load__<T>, ReturnScalar>;
    using base::var_name;
    using base::instance;
    static_assert(std::is_base_of<matrix_cl<ReturnScalar>,typename std::remove_reference_t<T>>::value, "load__: argument a must be a matrix_cl<T>!");
    static_assert(std::is_arithmetic<ReturnScalar>::value, "load__: T in matrix_cl<T> a argument must be an arithmetic type!");
    load__(T&& a) : a_(std::forward<T>(a)) {}

    kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const{
      if(generated.count(instance)==0) {
        generated.insert(instance);
        var_name = ng.generate();
        kernel_parts res;
        std::string type = type_str<ReturnScalar>::name;
        res.body = type + " " + var_name + " = 0;"
                   " if (!((!contains_nonzero(" + var_name + "_view, LOWER) && " + j + " < " + i + ") || (!contains_nonzero(" + var_name + "_view, UPPER) && " + j + " > " + i + "))) {"
                   + var_name + " = " + var_name + "_global[" + i + " + " + var_name + "_rows * " + j + "];}\n";
        res.args = "__global " + type + "* " + var_name + "_global, int " + var_name + "_rows, int " + var_name + "_view, ";
        return res;
      }
      else{
        return {};
      }
    }

    matrix_cl<ReturnScalar> eval() const{
      return a_;
    }

    void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const{
      if(generated.count(instance)==0) {
        generated.insert(instance);
        kernel.setArg(arg_num++, a_.buffer());
        kernel.setArg(arg_num++, a_.rows());
        kernel.setArg(arg_num++, a_.view());
      }
    }

    void add_event(cl::Event& e) const{
      a_.add_read_event(e);
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

private:
    T a_;
};

}
}

#endif
#endif
