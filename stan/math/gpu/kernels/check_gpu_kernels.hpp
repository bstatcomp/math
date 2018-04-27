#ifndef STAN_MATH_GPU_CHECK_GPU_KERNELS_HPP
#define STAN_MATH_GPU_CHECK_GPU_KERNELS_HPP

#include <string>

/**
 *  @file stan/math/gpu/basic_matrix_kernels.hpp
 *  @brief Kernel sources for basic matrix operations on the gpu:
 *    copy, copy lower/upper triangular, copy triangular transposed,
 *    copy submatrix, init to zeros, init to identity,
 *    add, subtract, transpose
 */

namespace stan {
  namespace math {
	
	std::string check_nan_kernel = 
	"#ifdef cl_khr_fp64 \n"
	"	#pragma OPENCL EXTENSION cl_khr_fp64 : enable \n"
	"#elif defined(cl_amd_fp64) \n"
	"	#pragma OPENCL EXTENSION cl_amd_fp64 : enable \n"
	"#else \n"
	"	#error \"Double not supported by OpenCL implementation.\" \n"
	"#endif \n"
	"#define A(i,j)  A[j*rows+i] \n"
	"__kernel void check_nan( \n"
	"			__global double *A, \n"
	"			int rows, \n"
	"			int cols, \n"
	"			__global int *flag) { \n"
	" const int i = get_global_id(0); \n"
	"  const int j = get_global_id(1); \n"
	"  if( i < rows && j < cols ) {  \n"
	"	if (isnan(A(i,j))) { \n"
	"	  flag[0] = 1; \n"
	"	} \n"
	" } \n"
	"} \n";

	std::string check_diagonal_zeros_kernel = 
	"__kernel void check_diagonal_zeros( \n"
	"			  __global double *A, \n"
	"			  int rows, \n"
	"			  int cols, \n"
	"			  __global int *flag) { \n"
	" const int i = get_global_id(0); \n"
	"  if( i < rows && i < cols ) {  \n"
	"	if (A(i,i) == 0) { \n"
	"	  flag[0] = 1; \n"
	"	} \n"
	"  } \n"
	"} \n";

	std::string check_symmetric_kernel = 
	"__kernel void check_symmetric( \n"
	"			__global double *A, \n"
	"			int rows, \n"
	"			int cols, \n"
	"			__global int *flag, \n"
	"			double tolerance) { \n"
	" const int i = get_global_id(0); \n"
	"  const int j = get_global_id(1); \n"
	" if( i < rows && j < cols ) {  \n"
	"	double diff = fabs(A(i,j)-A(j,i)); \n"
	"	if ( diff > tolerance ) { \n"
	"	  flag[0] = 1; \n"
	"	} \n"
	"  } \n"
	"} \n";

  }
}
#endif