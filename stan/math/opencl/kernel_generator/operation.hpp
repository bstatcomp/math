#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_OPERATION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_OPERATION_HPP
#ifdef STAN_OPENCL

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

class operation_base{};

template<typename Derived, typename ReturnScalar>
class operation : public operation_base{
public:
    static const int dynamic = -1;
    const std::string var_name;
    operation() : var_name(get_variable_name()){}

    Derived& derived() {
      return *static_cast<Derived*>(this);
    }

    const Derived& derived() const {
      return *static_cast<const Derived*>(this);
    }

    matrix_cl<ReturnScalar> eval() const{
      int n_rows = derived().rows();
      int n_cols = derived().cols();
      kernel_parts parts = derived().generate("i","j");
      std::string type = type_str<ReturnScalar>::name;
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
      derived().set_args(kernel,arg_num);

      matrix_cl<ReturnScalar> res(n_rows,n_cols, derived().view());
      kernel.setArg(arg_num++, res.buffer());
      kernel.setArg(arg_num++, n_rows);

      cl::Event e;
      opencl_context.queue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_rows,n_cols), cl::NullRange, nullptr, &e);
      derived().add_event(e);
      res.add_write_event(e);
      return res;
    }

    operator matrix_cl<ReturnScalar>(){
      return eval();
    }
};

}
}

#endif
#endif
