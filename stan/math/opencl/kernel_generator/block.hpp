#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_BLOCK_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_BLOCK_HPP
#ifdef  STAN_OPENCL

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

namespace stan{
namespace math{

template<typename T>
class block__ : public operation<block__<T>, typename std::remove_reference_t<T>::ReturnScalar> {
public:
  using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
  using base = operation<block__<T>, ReturnScalar>;
  using base::var_name;
  using base::instance;

  block__(T&& a, int start_row, int start_col, int rows, int cols) :
  a_(std::forward<T>(a)), start_row_(start_row), start_col_(start_col), rows_(rows), cols_(cols) {
    if ((a.rows() != base::dynamic && (start_row + rows) > a.rows()) ||
        (a.cols() != base::dynamic && (start_col + cols) > a.cols())) {
      domain_error("block", "block of \"a\"", " is out of bounds", "");
    }
  }

  inline kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const{
    kernel_parts res = a_.generate(ng, generated, "(" + i + " + " + std::to_string(start_row_) + ")", "(" + j + " + " + std::to_string(start_col_) + ")");
    var_name = a_.var_name;
    return res;
  }

  inline kernel_parts generate_lhs(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const{
    return a_.generate_lhs(ng, generated, "(" + i + " + " + std::to_string(start_row_) + ")", "(" + j + " + " + std::to_string(start_col_) + ")");
  }

  inline void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const{
    if(generated.count(instance)==0) {
      generated.insert(instance);
      a_.set_args(generated, kernel, arg_num);
    }
  }

  inline void add_event(cl::Event& e) const {
    a_.add_event(e);
  }

  inline void add_write_event(cl::Event& e) const {
    a_.add_event(e);
  }

  inline matrix_cl_view view() const {
    return transpose(a_.view());
  }

  inline int rows() const {
    return rows_;
  }

  inline int cols() const {
    return cols_;
  }

  template<typename T_expression, typename = enable_if_all_usable_as_operation<T>>
  const block__<T>& operator= (T_expression&& input) const{
    auto expression = as_operation(std::forward<T_expression>(input));
    expression.evaluate_into(*this);
    return *this;
  }

protected:
  T a_;
  int start_row_, start_col_, rows_, cols_;
};

template<typename T, typename = enable_if_all_usable_as_operation<T>>
inline block__<as_operation_t<T>> block(T&& a, int start_row, int start_col, int rows, int cols) {
  return block__<as_operation_t<T>>(as_operation(std::forward<T>(a)), start_row, start_col, rows, cols);
}

}
}

#endif
#endif
