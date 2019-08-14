#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_OPERATION_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_OPERATION_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/kernel_cl.hpp>
#include <CL/cl.hpp>
#include <string>
#include <map>
#include <set>

#include <iostream>

namespace stan {
namespace math {

struct kernel_parts{
    std::string body, args;
};

class operation_base{
public:
    operation_base() : instance(instance_counter++){}
protected:
    static int instance_counter;
    int instance;
};

int operation_base::instance_counter = 0;

template<typename Derived, typename ReturnScalar>
class operation : public operation_base{
public:
    static const int dynamic = -1;

    Derived& derived() {
      return *static_cast<Derived*>(this);
    }

    const Derived& derived() const {
      return *static_cast<const Derived*>(this);
    }

    matrix_cl<ReturnScalar> eval() const {
      int n_rows = derived().rows();
      int n_cols = derived().cols();
      name_generator ng;
      std::set<int> generated;
      kernel_parts parts = derived().generate(ng, generated,"i","j");
      std::string type = type_str<ReturnScalar>::name;
      std::string src = "kernel void calculate(" + parts.args + "__global " + type + "* output, int output_rows){\n"
        "int i = get_global_id(0);"
        "int j = get_global_id(1);\n"
        + parts.body +
        "output[i + output_rows * j] = " + var_name + ";}";
      matrix_cl<ReturnScalar> res(n_rows,n_cols, derived().view());
      try {
        if(kernel_cache.count(src)==0){
          std::cout << src << std::endl;
          auto opts = opencl_context.base_opts();
          kernel_cache[src] = opencl_kernels::compile_kernel("calculate", {view_kernel_helpers, src.c_str()}, opts);
        }
        cl::Kernel& kernel = kernel_cache[src];
        int arg_num = 0;
        generated.clear();
        derived().set_args(generated,kernel,arg_num);

        kernel.setArg(arg_num++, res.buffer());
        kernel.setArg(arg_num++, n_rows);

        cl::Event e;
        opencl_context.queue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_rows,n_cols), cl::NullRange, nullptr, &e);
        derived().add_event(e);
        res.add_write_event(e);
      }
      catch(cl::Error e){
        check_opencl_error("operation.eval()", e);
      }
      return res;
    }

    operator matrix_cl<ReturnScalar>() const {
      return derived().eval();
    }

protected:
    mutable std::string var_name;
    static std::map<std::string, cl::Kernel> kernel_cache;
};

template<typename Derived, typename ReturnScalar>
std::map<std::string, cl::Kernel> operation<Derived,ReturnScalar>::kernel_cache;

}
}

#endif
#endif
