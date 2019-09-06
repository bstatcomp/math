#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_TRANSPOSE_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_TRANSPOSE_HPP
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

namespace stan {
namespace math {

template<typename T>
class transpose__ : public operation<transpose__<T>, typename std::remove_reference_t<T>::ReturnScalar> {
public:
  using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
  using base = operation<transpose__<T>, ReturnScalar>;
  using base::var_name;
  using base::instance;

  explicit transpose__(T&& a) : a_(std::forward<T>(a)) {}

  inline kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const {
    kernel_parts res = a_.generate(ng, generated, j, i);
    var_name = a_.var_name;
    return res;
  }

  inline void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const {
    if (generated.count(instance) == 0) {
      generated.insert(instance);
      a_.set_args(generated, kernel, arg_num);
    }
  }

  inline void add_event(cl::Event& e) const {
    a_.add_event(e);
  }

  inline matrix_cl_view view() const {
    return transpose(a_.view());
  }

  inline int rows() const {
    return a_.cols();
  }

  inline int cols() const {
    return a_.rows();
  }

protected:
  T a_;
};

template<typename T, typename = enable_if_none_arithmetic_all_usable_as_operation<T>>
inline transpose__<as_operation_t<T>> transpose(T&& a) {
  return transpose__<as_operation_t<T>>(as_operation(std::forward<T>(a)));
}

}
}

#endif
#endif
