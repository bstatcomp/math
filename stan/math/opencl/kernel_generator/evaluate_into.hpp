#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_EVALUATE_INTO_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_EVALUATE_INTO_HPP

#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <stan/math/opencl/kernel_generator/as_operation.hpp>
#include <stan/math/opencl/kernel_generator/is_valid_expression.hpp>
#include <utility>

namespace stan{
namespace math{

template<typename Derived, typename ReturnScalar>
template<typename T_lhs>
void operation<Derived, ReturnScalar>::evaluate_into(T_lhs&& lhs) const {
  using enable = enable_if_all_valid_expressions<T_lhs>;
  using cache = operation<Derived, ReturnScalar>::cache<T_lhs>;
  auto lhs_expression = as_operation(std::forward<T_lhs>(lhs));

  int n_rows = derived().rows();
  int n_cols = derived().cols();
  const char* function = "evaluate_into";
  if(n_rows!=dynamic) {
    check_size_match(function, "Rows of ", "*this", n_rows, "rows of ", "lhs_expression", lhs_expression.rows());
  }
  if(n_cols!=dynamic) {
    check_size_match(function, "Columns of ", "*this", n_cols, "columns of ", "lhs_expression", lhs_expression.cols());
  }
  try {
    std::set<int> generated;
    if(cache::kernel() == NULL){
      name_generator ng;
      kernel_parts parts = derived().generate(generated, ng,"i","j");
      kernel_parts out_parts = lhs_expression.generate_lhs(generated, ng, "i","j");
      std::string src = "kernel void calculate(" + parts.args + out_parts.args.substr(0,out_parts.args.size()-2) + "){\n"
                       "int i = get_global_id(0);"
                       "int j = get_global_id(1);\n"
                        + parts.body +
                        out_parts.body + " = " + var_name + ";}";
      auto opts = opencl_context.base_opts();
      cache::kernel = opencl_kernels::compile_kernel("calculate", {view_kernel_helpers, src.c_str()}, opts);
      generated.clear();
    }
    int arg_num = 0;
    derived().set_args(generated,cache::kernel,arg_num);
    lhs_expression.set_args(generated, cache::kernel, arg_num);

    cl::Event e;
    opencl_context.queue().enqueueNDRangeKernel(cache::kernel, cl::NullRange, cl::NDRange(n_rows,n_cols), cl::NullRange, nullptr, &e);
    derived().add_event(e);
    lhs_expression.add_write_event(e);
  }
  catch(cl::Error e){
    check_opencl_error("operation.evaluate_into", e);
  }
}

}
}


#endif
