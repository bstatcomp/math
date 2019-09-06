#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_MULTI_RESULT_KERNEL_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_MULTI_RESULT_KERNEL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/is_valid_expression.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <string>
#include <set>

namespace stan{
namespace math{

static kernel_parts multi_result_kernel_generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j){
  return {"",""};
}

template<typename T_res0, typename... T_expressions>
static kernel_parts multi_result_kernel_generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j, T_res0 expression0, T_expressions... expressions){
  kernel_parts parts0 = expression0.generate(ng, generated, i, j);
  kernel_parts parts = multi_result_kernel_generate(ng, generated, i, j, expressions...);
  return {parts0.body + parts.body,parts0.args + parts.args};
}

template<typename... T_expressions, typename = enable_if_all_usable_as_operation<T_expressions...>>
static std::tuple<matrix_cl<T_expressions::ResultScalar>...> multi_result_kernel(T_expressions... expressions){
  int rows = std::max(expressions.rows()...);
  int cols = std::max(expressions.cols()...);

  name_generator ng;
  std::set<int> generated;
}

}
}

#endif
#endif
