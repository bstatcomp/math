#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_OPERATION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_OPERATION_HPP

#include <stan/math/opencl/kernel_generator/utility.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_cl.hpp>
#include <CL/cl.hpp>
#include <string>

#include <iostream>

namespace stan {
namespace math {

struct kernel_parts{
    std::string body, args;
};

template<typename T>
class operation{
public:
    const std::string var_name;
    operation() : var_name(get_variable_name()){}

    virtual kernel_parts generate(const std::string& i, const std::string& j) const = 0;

    virtual int rows() const = 0;

    virtual int cols() const = 0;

    virtual matrix_cl_view view() const = 0;

    virtual void set_args(cl::Kernel& kernel, int& arg_num) const = 0;

    virtual void add_event(cl::Event& e) const = 0;

    virtual matrix_cl<T> eval(){
      int n_rows = rows();
      int n_cols = cols();
      kernel_parts parts = generate("i","j");
      std::string type = type_str<T>::name;
      std::string src = "kernel void calculate(" + parts.args + "__global " + type + "* output, int output_rows){\n"
        "int i = get_global_id(0);"
        "int j = get_global_id(1);\n"
        + parts.body +
        "output[i + output_rows * j] = " + var_name + ";}";
      std::cout << src << std::endl;
      //TODO: cache kernel instead of recompiling!
      auto opts = opencl_context.base_opts();
      cl::Kernel kernel = opencl_kernels::compile_kernel("calculate", {view_kernel_helpers, src.c_str()}, opts);
      int arg_num = 0;
      set_args(kernel,arg_num);

      matrix_cl<T> res(n_rows,n_cols, view());
      kernel.setArg(arg_num++, res.buffer());
      kernel.setArg(arg_num++, n_rows);

      cl::Event e;
      opencl_context.queue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_rows,n_cols), cl::NullRange, nullptr, &e);
      add_event(e);
      res.add_write_event(e);
      return res;
    }

    operator matrix_cl<T>(){
      return eval();
    }
};

}
}

#endif //STAN_MATH_OPENCL_KERNEL_GENERATOR_OPERATOR_HPP
