#ifndef STAN_MATH_GPU_EVENT_UTILS_HPP
#define STAN_MATH_GPU_EVENT_UTILS_HPP
#ifdef STAN_OPENCL
#include <CL/cl.hpp>

namespace stan {
namespace math {
  //TODO(Steve): Use variadic templating here to have arbitrary lists.
  std::vector<cl::Event> event_concat_cl(const std::vector<cl::Event>& v1, const std::vector<cl::Event>& v2) {
      std::vector<cl::Event> vec_concat;
      vec_concat.insert(vec_concat.end(), v1.begin(), v1.end());
      vec_concat.insert(vec_concat.end(), v2.begin(), v2.end());
      return vec_concat;   // rvo blocked
  }
}
}

#endif
#endif
