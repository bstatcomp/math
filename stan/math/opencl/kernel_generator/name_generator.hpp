#ifndef STAN_MATH_OPENCL_KERNEL_GENERATOR_NAME_GENERATOR_HPP
#define STAN_MATH_OPENCL_KERNEL_GENERATOR_NAME_GENERATOR_HPP
#ifdef STAN_OPENCL

#include <string>

namespace stan {
namespace math {

class name_generator {
public:
  name_generator() : i_(0) {}

  inline std::string generate() {
    return "var" + std::to_string(++i_);
  }

private:
  int i_;
};

}
}

#endif
#endif
