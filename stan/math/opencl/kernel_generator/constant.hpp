#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_CONSTANT_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_CONSTANT_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/matrix_cl_view.hpp>
#include <stan/math/opencl/kernel_generator/type_str.hpp>
#include <stan/math/opencl/kernel_generator/name_generator.hpp>
#include <stan/math/opencl/kernel_generator/operation.hpp>
#include <string>
#include <type_traits>
#include <set>

namespace stan {
namespace math {


template<typename T>
class constant__ : public operation<constant__<T>, T>{
public:
    static_assert(std::is_arithmetic<T>::value, "std::is_arithmetic<T> must be true for constants!");
    using ReturnScalar = T;
    using base = operation<constant__<T>, T>;
    using base::var_name;
    using base::instance;

    constant__(const T a) : a_(a) {}

    kernel_parts generate(name_generator& ng, std::set<int>& generated, const std::string& i, const std::string& j) const{
      if(generated.count(instance)==0) {
        generated.insert(instance);
        var_name = ng.generate();
        kernel_parts res;
        std::string type = type_str<T>::name;
        res.body = type + " " + var_name + " = " + std::to_string(a_) + ";\n";
        return res;
      }
      else{
        return {};
      }
    }

    void set_args(std::set<int>& generated, cl::Kernel& kernel, int& arg_num) const{

    }

    void add_event(cl::Event& e) const{

    }

    int rows() const{
      return base::dynamic;
    }

    int cols() const{
      return base::dynamic;
    }

    matrix_cl_view view() const{
      return matrix_cl_view::Entire;
    }

private:
    const T a_;
};

}
}

#endif
#endif
