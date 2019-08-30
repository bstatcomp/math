#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_BROADCAST_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_BROADCAST_HPP
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

template<typename T, bool Rows, bool Cols>
class broadcast__ : public operation<broadcast__<T, Rows, Cols>, typename std::remove_reference_t<T>::ReturnScalar> {
public:
    using ReturnScalar = typename std::remove_reference_t<T>::ReturnScalar;
    using base = operation<broadcast__<T, Rows, Cols>, ReturnScalar>;
    using base::var_name;
    using base::instance;

    broadcast__(T&& a) : a_(std::forward<T>(a)) {
      const char* function = "broadcast";
      if (Rows) {
        check_size_match(function, "Rows of ", "a", a.rows(), "", "1", 1);
      }
      if (Cols) {
        check_size_match(function, "Columns of ", "a", a.cols(), "", "1", 1);
      }
    }

    kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const {
      kernel_parts res = a_.generate(ng, generated, Rows ? "0" : i, Cols ? "0" : j);
      var_name = a_.var_name;
      return res;
    }

    void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const {
      a_.set_args(generated, kernel, arg_num);
    }

    void add_event(cl::Event& e) const {
      a_.add_event(e);
    }

    int rows() const {
      return Rows ? base::dynamic : a_.rows();
    }

    int cols() const {
      return Cols ? base::dynamic : a_.cols();
    }

    matrix_cl_view view() const {
      matrix_cl_view view = a_.view();
      if (Rows) {
        view = either(view, matrix_cl_view::Lower);
      }
      if (Cols) {
        view = either(view, matrix_cl_view::Upper);
      }
      return view;
    }

protected:
    const T a_;
};

template<bool Rows, bool Cols, typename T, typename Cond = enable_if_none_arithmetic_all_usable_as_operation<T>>
auto broadcast(T&& a) -> broadcast__<decltype(as_operation(std::forward<T>(a))), Rows, Cols>{
  return broadcast__<decltype(as_operation(std::forward<T>(a))), Rows, Cols>(as_operation(std::forward<T>(a)));
}

}
}
#endif
#endif
