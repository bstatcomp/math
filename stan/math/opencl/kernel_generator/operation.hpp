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

    matrix_cl<ReturnScalar> eval() const {
      matrix_cl<ReturnScalar> res(derived().rows(), derived().cols(), derived().view());
      this->evaluate_into(res);
      return res;
    }

    operator matrix_cl<ReturnScalar>() const {
      return derived().eval();
    }

    template<typename T_lhs>
    void evaluate_into(T_lhs&& lhs) const;

protected:
    mutable std::string var_name;
    static std::map<std::string, cl::Kernel> kernel_cache;

    Derived& derived() {
      return *static_cast<Derived*>(this);
    }

    const Derived& derived() const {
      return *static_cast<const Derived*>(this);
    }
};

template<typename Derived, typename ReturnScalar>
std::map<std::string, cl::Kernel> operation<Derived,ReturnScalar>::kernel_cache;

}
}

#endif
#endif

#include <stan/math/opencl/kernel_generator/evaluate_into.hpp>