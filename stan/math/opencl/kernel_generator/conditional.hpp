#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_CONDITIONAL_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_CONDITIONAL_HPP
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

template<typename T_condition, typename T_then, typename T_else>
class conditional__ : public operation<conditional__<T_condition, T_then, T_else>, typename std::common_type<typename std::remove_reference_t<T_then>::ReturnScalar, typename std::remove_reference_t<T_else>::ReturnScalar>::type> {
public:
    using ReturnScalar = typename std::common_type<typename std::remove_reference_t<T_then>::ReturnScalar, typename std::remove_reference_t<T_else>::ReturnScalar>::type;
    using base = operation<conditional__<T_condition, T_then, T_else>, ReturnScalar>;
    using base::var_name;
    using base::instance;

    conditional__(T_condition&& condition, T_then&& then, T_else&& els) :
        condition_(std::forward<T_condition>(condition)), then_(std::forward<T_then>(then)), else_(std::forward<T_else>(els)) {
      if(condition.rows()!=base::dynamic && then.rows()!=base::dynamic) {
        check_size_match("conditional", "Rows of ", "condition", condition.rows(), "rows of ", "then", then.rows());
      }
      if(condition.cols()!=base::dynamic && then.cols()!=base::dynamic) {
        check_size_match("conditional", "Columns of ", "condition", condition.cols(), "columns of ", "then", then.cols());
      }

      if(condition.rows()!=base::dynamic && els.rows()!=base::dynamic) {
        check_size_match("conditional", "Rows of ", "condition", condition.rows(), "rows of ", "else", els.rows());
      }
      if(condition.cols()!=base::dynamic && els.cols()!=base::dynamic) {
        check_size_match("conditional", "Columns of ", "condition", condition.cols(), "columns of ", "else", els.cols());
      }
    }

    kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const {
      if(generated.count(instance)==0) {
        kernel_parts condition_parts = condition_.generate(ng, generated, i, j);
        kernel_parts then_parts = then_.generate(ng, generated, i, j);
        kernel_parts else_parts = else_.generate(ng, generated, i, j);
        generated.insert(instance);
        var_name = ng.generate();
        kernel_parts res;
        res.body = condition_parts.body + then_parts.body + else_parts.body + type_str<ReturnScalar>::name + " " + var_name + " = " + condition_.var_name + " ? " + then_.var_name + " : " + else_.var_name + ";\n";
        res.args = condition_parts.args + then_parts.args + else_parts.args;
        return res;
      }
      else{
        return {};
      }
    }

    void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const {
      condition_.set_args(generated, kernel, arg_num);
      then_.set_args(generated, kernel, arg_num);
      else_.set_args(generated, kernel, arg_num);
    }

    void add_event(cl::Event& e) const {
      condition_.add_event(e);
      then_.add_event(e);
      else_.add_event(e);
    }

    int rows() const {
      return condition_.rows();
    }

    int cols() const {
      return condition_.cols();
    }

    matrix_cl_view view() const {
      matrix_cl_view condition_view = condition_.view();
      matrix_cl_view then_view = then_.view();
      matrix_cl_view else_view = else_.view();
      return both(either(then_view,else_view), both(condition_view,then_view));
    }

protected:
    T_condition condition_;
    T_then then_;
    T_else else_;
};

template<typename T_condition, typename T_then, typename T_else, typename = enable_if_all_usable_as_operation<T_condition, T_then, T_else>>
conditional__<as_operation_t<T_condition>, as_operation_t<T_then>, as_operation_t<T_else>> conditional(T_condition&& condition, T_then&& then, T_else&& els) {
  return {as_operation(std::forward<T_condition>(condition)),
          as_operation(std::forward<T_then>(then)),
          as_operation(std::forward<T_else>(els))};
}

}
}
#endif
#endif
