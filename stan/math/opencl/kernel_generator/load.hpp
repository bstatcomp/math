#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_LOAD_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_LOAD_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/kernel_generator/utility.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <string>

namespace stan {
namespace math {

template<typename T>
class load__ : public operation<load__<T>, T>{
public:
    using ReturnScalar = T;
    using operation<load__<T>, T>::var_name;
    load__(const matrix_cl<T>& a) : a_(a) {}

    kernel_parts generate(const std::string& i, const std::string& j) const{
      kernel_parts res;
      std::string type = type_str<T>::name;
      res.body = type + " " + var_name + " = 0;"
        " if (!((!contains_nonzero(" + var_name + "_view, LOWER) && j < i) || (!contains_nonzero(" + var_name + "_view, UPPER) && j > i))) {"
        + var_name + " = " + var_name + "_global[" + i + " + " + var_name + "_rows * " + j + "];}\n";
      res.args = "__global " + type + "* " + var_name + "_global, int " + var_name + "_rows, int " + var_name + "_view, ";
      return res;
    }

    void set_args(cl::Kernel& kernel, int& arg_num) const{
      kernel.setArg(arg_num++, a_.buffer());
      kernel.setArg(arg_num++, a_.rows());
      kernel.setArg(arg_num++, a_.view());
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
    const matrix_cl<T>& a_;
};

}
}

#endif
#endif
