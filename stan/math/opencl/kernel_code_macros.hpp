#ifndef STAN_MATH_OPENCL_KERNEL_CODE_MACROS_HPP
#define STAN_MATH_OPENCL_KERNEL_CODE_MACROS_HPP
#ifdef STAN_OPENCL
#include <boost/algorithm/string/replace.hpp>
#include <string>

// Used for importing the OpenCL kernels at compile time.
// There has been much discussion about the best ways to do this:
// https://github.com/bstatcomp/math/pull/7
// and https://github.com/stan-dev/math/pull/966
#ifndef STRINGIFY
#define STRINGIFY(...) #__VA_ARGS__
#endif

#ifndef COMMON_CODE
#define COMMON_CODE(...) boost::replace_all_copy(std::string(STRINGIFY(__VA_ARGS__)),"enum class", "enum"); __VA_ARGS__
#endif


#endif
#endif
