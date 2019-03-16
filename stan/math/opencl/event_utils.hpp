#ifndef STAN_MATH_GPU_EVENT_UTILS_HPP
#define STAN_MATH_GPU_EVENT_UTILS_HPP
#ifdef STAN_OPENCL

#include <stan/math/opencl/matrix_cl.hpp>
#include <type_traits>
#include <CL/cl.hpp>

namespace stan {
namespace math {
  // Ends the recursion
  std::vector<cl::Event> event_concat_cl(const std::vector<cl::Event>& v1) {
    return v1;
  }
  // Ends the recursion
  std::vector<cl::Event> event_concat_cl(const matrix_cl& A) {
    return A.events();
  }

  // If there is one non cl events
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
  std::vector<cl::Event> event_concat_cl(const std::vector<cl::Event>& v1, const T& throwaway_val) {
    return v1;
  }
  // If both are non cl events return an empty vector
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
  std::vector<cl::Event> event_concat_cl(const T& v1, const T& throwaway_val) {
    std::vector<cl::Event> vec_concat;
    return vec_concat;
  }
  // end is non cl event return an empty vector
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value, int> = 0>
  std::vector<cl::Event> event_concat_cl(const T& throwaway_val) {
    std::vector<cl::Event> vec_concat;
    return vec_concat;
  }
  // Put two events together
  std::vector<cl::Event> event_concat_cl(const std::vector<cl::Event>& v1, const std::vector<cl::Event>& v2) {
    std::vector<cl::Event> vec_concat;
    vec_concat.insert(vec_concat.end(), v1.begin(), v1.end());
    vec_concat.insert(vec_concat.end(), v2.begin(), v2.end());
    return vec_concat;
  }
  // Concat event and matrix
  std::vector<cl::Event> event_concat_cl(const std::vector<cl::Event>& v1, const matrix_cl& B) {
    return event_concat_cl(v1, B.events());
  }
  // If we have an matrix_cl and whatever
  template <typename ...Args>
  std::vector<cl::Event> event_concat_cl(const matrix_cl& A, Args... args) {
    return event_concat_cl(A.events(), args...);
  }
  // If we have an event vector and whatever
  template <typename ...Args>
  std::vector<cl::Event> event_concat_cl(const std::vector<cl::Event>& v1, Args... args) {
    std::vector<cl::Event> vec_concat = event_concat_cl(args...);
    vec_concat.insert(vec_concat.end(), v1.begin(), v1.end());
    return vec_concat;
  }

}
}

#endif
#endif
