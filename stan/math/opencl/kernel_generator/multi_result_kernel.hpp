#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_MULTI_RESULT_KERNEL_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_MULTI_RESULT_KERNEL_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/kernel_generator/is_valid_expression.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/kernel_generator/as_operation.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <string>
#include <tuple>
#include <utility>
#include <set>

namespace stan{
namespace math{

namespace internal{

/**
 * A perfect forwarding wrapper. Needed since references normally can not be used in tuples.
 */
template<typename T>
struct wrapper{
  T x;
  explicit wrapper(T&& x) : x(std::forward<T>(x)){}
};

template<typename T>
wrapper<T> make_wrapper(T&& x){
  return wrapper<T>(std::forward<T>(x));
}

// Pack can only be at the end of the list. We need 2 packs, so we nest structs.
template<int n, typename... T_results>
struct multi_result_kernel_internal{
  template <typename... T_expressions>
  struct inner {
    static cl::Kernel kernel_;
    using next = typename multi_result_kernel_internal<n - 1, T_results...>::template inner<T_expressions...>;
    static kernel_parts generate(std::set<int>& generated, name_generator& ng,
                                 const std::string& i, const std::string& j,
                                 std::tuple<wrapper<T_results>...>& results,
                                 std::tuple<wrapper<T_expressions>...>& expressions) {
      kernel_parts parts = next::generate(generated, ng, i, j, results, expressions);

      kernel_parts parts0 = std::get<n>(expressions).x.generate(generated, ng, i, j);
      kernel_parts out_parts0
          = std::get<n>(results).x.generate_lhs(generated, ng, i, j);

      parts0.args += out_parts0.args;
      parts0.body += out_parts0.body + " = " + std::get<n>(expressions).x.var_name + ";\n";

      return {parts.body + parts0.body, parts.args + parts0.args};
    }

    static void set_args(std::set<int>& generated, cl::Kernel& kernel,
                         int& arg_num, std::tuple<wrapper<T_results>...>& results,
                         std::tuple<wrapper<T_expressions>...>& expressions) {
      next::set_args(generated, kernel, arg_num, results, expressions);

      std::get<n>(expressions).x.set_args(generated, kernel, arg_num);
      std::get<n>(results).x.set_args(generated, kernel, arg_num);
    }

    static void add_event(cl::Event e, std::tuple<wrapper<T_results>...>& results,
                          std::tuple<wrapper<T_expressions>...>& expressions) {
      next::add_event(e, results, expressions);

      std::get<n>(expressions).x.add_event(e);
      std::get<n>(results).x.add_write_event(e);
    }
  };
};

template<int n, typename... T_results>
template <typename... T_expressions>
cl::Kernel multi_result_kernel_internal<n, T_results...>::inner<T_expressions...>::kernel_;

template<typename... T_results>
struct multi_result_kernel_internal<-1, T_results...>{
  template <typename... T_expressions>
  struct inner {
    static kernel_parts generate(std::set<int>& generated, name_generator& ng,
                                 const std::string& i, const std::string& j,
                                 std::tuple<wrapper<T_results>...>& results,
                                 std::tuple<wrapper<T_expressions>...>& expressions) {
      return {"", ""};
    }

    static void set_args(std::set<int>& generated, cl::Kernel& kernel,
                         int& arg_num, std::tuple<wrapper<T_results>...>& results,
                         std::tuple<wrapper<T_expressions>...>& expressions) {}

    static void add_event(cl::Event e, std::tuple<wrapper<T_results>...>& results,
                          std::tuple<wrapper<T_expressions>...>& expressions) {}
  };
};

} // namespace internal

/**
 * Represents multiple expressions that will be used in same kernel.
 * @tparam T_expressions types of expressions
 */
template<typename... T_expressions>
class expressions__{
 public:
  /**
   * Constructor.
   * @param expressions expressions that will be used in same kernel.
   */
  explicit expressions__(T_expressions&&... expressions) :
      expressions_(internal::wrapper<T_expressions>(std::forward<T_expressions>(expressions))...){}
 private:
  std::tuple<internal::wrapper<T_expressions>...> expressions_;
  template<typename... T_results>
  friend class results__;
};

/**
 * Deduces types for constructing \c expressions__ object.
 * @tparam T_expressions types of expressions
 * @param expressions expressions that will be used in same kernel.
 */
template<typename... T_expressions>
expressions__<T_expressions...> expressions(T_expressions&&... expressions){
  return expressions__<T_expressions...>(std::forward<T_expressions>(expressions)...);
}

/**
 * Represents results that will be calculated in same kernel.
 * @tparam T_results types of results
 */
template<typename... T_results>
class results__{
 public:
  /**
   * Constructor.
   * @param results results that will be calculated in same kernel
   */
  explicit results__(T_results&... results) : results_(&results...){}

  /**
   * Assigning \c expressions__ object to \c results__ object generates and
   * executes the kernel that evaluates expressions and stores them into results this object consists of..
   * @tparam T_expressions types of expressions
   * @param expressions expressions
   */
  template<typename... T_expressions, typename = std::enable_if_t<sizeof...(T_results) == sizeof...(T_expressions)>>
  void operator=(expressions__<T_expressions...> expressions){
    assignment(expressions, std::make_index_sequence<sizeof...(T_expressions)>{});
  }
private:
  std::tuple<T_results*...> results_;
  /**
   * Assignment of expressions to results.
   * @tparam T_expressions types of expressions
   * @param exprs expressions
   */
  template<typename... T_expressions, size_t... Is>
  void assignment(expressions__<T_expressions...> exprs, std::index_sequence<Is...>){
    auto expressions = std::make_tuple(internal::make_wrapper(std::forward<decltype(as_operation(std::get<Is>(exprs.expressions_).x))>(as_operation(std::get<Is>(exprs.expressions_).x)))...);
    auto results = std::make_tuple(internal::make_wrapper(std::forward<decltype(as_operation(*std::get<Is>(results_)))>(as_operation(*std::get<Is>(results_))))...);
    assignment_impl(results, expressions);
  }

  /**
   * Implementation of assignments of expressions to results
   * @tparam T_res types of results
   * @tparam T_expressions types of expressions
   * @param results results
   * @param expressions expressions
   */
  template<typename... T_res, typename... T_expressions>
  static void assignment_impl(std::tuple<internal::wrapper<T_res>...>& results, std::tuple<internal::wrapper<T_expressions>...>& expressions){
    using T_First_Expr = typename std::remove_reference_t<std::tuple_element_t<0,std::tuple<T_expressions...>>>;
    using impl = typename internal::multi_result_kernel_internal<std::tuple_size<std::tuple<T_expressions...>>::value-1, T_res...>:: template inner<T_expressions...>;

    int n_rows = std::get<0>(expressions).x.rows();
    int n_cols = std::get<0>(expressions).x.cols(); //TODO fix rows/cols

    name_generator ng;
    std::set<int> generated;

    try {
      if(impl::kernel_() == NULL){
        kernel_parts parts = impl::generate(generated, ng, "i", "j", results, expressions);

        std::string src = "kernel void calculate(" + parts.args.substr(0,parts.args.size()-2) + "){\n"
                          "int i = get_global_id(0);"
                          "int j = get_global_id(1);\n"
                          + parts.body + "}";
        auto opts = opencl_context.base_opts();
        impl::kernel_ = opencl_kernels::compile_kernel("calculate", {view_kernel_helpers, src}, opts);
      }
      cl::Kernel& kernel = impl::kernel_;
      int arg_num = 0;
      generated.clear();

      impl::set_args(generated, kernel, arg_num, results, expressions);

      cl::Event e;
      opencl_context.queue().enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n_rows,n_cols), cl::NullRange, nullptr, &e);
      impl::add_event(e, results, expressions);
    }
    catch(cl::Error e){
      check_opencl_error("result__.assignment", e);
    }
  }
};

/**
 * Deduces types for constructing \c results__ object.
 * @tparam T_results types of results
 * @param results results that will be calculated in same kernel.
 */
template<typename... T_results>
results__<T_results...> results(T_results&... results){
  return results__<T_results...>(results...);
}

}
}

#endif
#endif
