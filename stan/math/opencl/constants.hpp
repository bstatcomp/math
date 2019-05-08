#ifndef STAN_MATH_OPENCL_CONSTANTS_HPP
#define STAN_MATH_OPENCL_CONSTANTS_HPP
#ifdef STAN_OPENCL
#include<stan/math/opencl/kernel_code_macros.hpp>

namespace stan {
namespace math {
// \cond
static const char* enums_kernel_code = COMMON_CODE(
        // \endcond
        enum class TriangularViewCL { Lower = 0, Upper = 1, Entire = 2 };
        enum class TriangularMapCL { UpperToLower = 0, LowerToUpper = 1 };
        // \cond
        );
// \endcond

}  // namespace math
}  // namespace stan
#endif
#endif
