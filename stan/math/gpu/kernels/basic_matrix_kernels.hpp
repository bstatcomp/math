#ifndef STAN_MATH_GPU_BASIC_MATRIX_KERNELS_HPP
#define STAN_MATH_GPU_BASIC_MATRIX_KERNELS_HPP

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

  std::string copy_matrix_kernel =
  "#define A(i,j)  A[j*rows+i] \n"
   "#define AT(i,j)  A[j*cols+i] \n"
  "#define B(i,j)  B[j*rows+i] \n"
  "#define BT(i,j)  B[j*cols+i] \n"
  "#define C(i,j)  C[j*rows+i] \n"
  "#ifdef cl_khr_fp64 \n"
  "  #pragma OPENCL EXTENSION cl_khr_fp64 : enable \n"
  "#elif defined(cl_amd_fp64) \n"
  "  #pragma OPENCL EXTENSION cl_amd_fp64 : enable \n"
  "#else \n"
  "  #error \"Double not supported by OpenCL implementation.\" \n"
  "#endif \n"
  " __kernel void copy( \n"
  "      __global double *A, \n"
  "      __global double *B, \n"
  "      unsigned int rows, \n"
  "      unsigned int cols) { \n"
  "  int i = get_global_id(0); \n"
  "  int j = get_global_id(1); \n"
  "  if ( i < rows && j < cols ) { \n"
  "   B(i,j) = A(i,j); \n"
  "  }\n"
  "}\n";

  std::string transpose_matrix_kernel =
  "__kernel void transpose( \n"
    "      __global double *B, \n"
    "      __global double *A, \n"
    "      int rows, \n"
    "      int cols ) { \n"
    " int i = get_global_id(0); \n"
    " int j = get_global_id(1); \n"
    " if ( i < rows && j < cols ) { \n"
    "  BT(j,i) = A(i,j);      \n"
    " } \n"
  "}\n";

  std::string zeros_matrix_kernel =
  "__kernel void zeros( \n"
    "      __global double *A, \n"
    "      unsigned int rows, \n"
    "      unsigned int cols, \n"
    "      unsigned int part) { \n"
  "  int i = get_global_id(0); \n"
  "  int j = get_global_id(1); \n"
  "  if ( i < rows && j < cols ) { \n"
  "   if ( part == 0 && j < i ) { \n"
  "     A(i,j) = 0; \n"
  "   } else if ( part == 1 && j > i ) { \n"
  "     A(i,j) = 0; \n"
  "   } else if ( part == 2 ) { \n"
  "     A(i,j) = 0; \n"
  "   } \n"
  "  } \n"
  "}\n";

  std::string identity_matrix_kernel =
  "__kernel void identity( \n"
  "      __global double *A, \n"
  "      unsigned int rows, \n"
  "      unsigned int cols) { \n"
  "  int i = get_global_id(0); \n"
  "  int j = get_global_id(1); \n"
  "  if ( i < rows && j < cols ) { \n"
  "     if ( i == j ) { \n"
  "     A(i,j) = 1.0; \n"
  "     } else { \n"
  "     A(i,j) = 0.0; \n"
  "     } \n"
  "  } \n"
  "}\n";

  std::string copy_triangular_matrix_kernel =
  "__kernel void copy_triangular( \n"
  "      __global double *A, \n"
  "      __global double *B, \n"
  "      unsigned int rows, \n"
  "      unsigned int cols, \n"
  "      unsigned int lower_upper) { \n"
  "  int i = get_global_id(0); \n"
  "  int j = get_global_id(1); \n"
  "  if ( i < rows && j < cols ) { \n"
  "   if ( !lower_upper && j <= i ) { \n"
  "     A(i,j) = B(i,j); \n"
  "   } else if ( !lower_upper ) { \n"
  "     A(i,j) = 0; \n"
  "   } else if ( lower_upper && j >= i ) { \n"
  "     A(i,j) = B(i,j); \n"
  "   } else if ( lower_upper && j < i ) { \n"
  "     A(i,j) = 0; \n"
  "   } \n"
  "  } \n"
  "}\n";

  std::string copy_triangular_transposed_matrix_kernel =
  "__kernel void copy_triangular_transposed( \n"
  "      __global double *A, \n"
  "      unsigned int rows, \n"
  "      unsigned int cols, \n"
  "      unsigned int lower_to_upper) { \n"
  "  int i = get_global_id(0); \n"
  "  int j = get_global_id(1); \n"
  "  if ( i < rows && j < cols ) { \n"
  "   if ( lower_to_upper && j > i ) { \n"
  "     AT(j,i) = A(i,j); \n"
  "   } else if ( !lower_to_upper && j > i ) { \n"
  "     A(i,j) = AT(j,i); \n"
  "   } \n"
  "  } \n"
  "}\n";

  std::string add_matrix_kernel =
  "__kernel void add( \n"
  "      __global double *C, \n"
  "      __global double *A, \n"
  "      __global double *B, \n"
  "      unsigned int rows, \n"
  "      unsigned int cols) { \n"
  "  int i = get_global_id(0); \n"
  "  int j = get_global_id(1); \n"
  "  if ( i < rows && j < cols ) { \n"
  "    C(i,j) = A(i,j) + B(i,j); \n"
  "  } \n"
  "}\n";

  std::string subtract_matrix_kernel =
  "__kernel void subtract(\n"
  "      __global double *C,\n"
  "      __global double *A,\n"
  "      __global double *B,\n"
  "      unsigned int rows,\n"
  "      unsigned int cols) {\n"
  "  int i = get_global_id(0);\n"
  "  int j = get_global_id(1);\n"
  "  if ( i < rows && j < cols ) {\n"
  "   C(i,j) = A(i,j)-B(i,j);\n"
  "  }\n"
  "}\n";

  std::string copy_submatrix_kernel =
  "#define src(i,j) src[j*src_rows+i] \n"
  "#define dst(i,j) dst[j*dst_rows+i] \n"
  "__kernel void copy_submatrix( \n"
  "      __global double *src, \n"
  "      __global double *dst, \n"
  "      unsigned int src_offset_i, \n"
  "      unsigned int src_offset_j, \n"
  "      unsigned int dst_offset_i, \n"
  "      unsigned int dst_offset_j, \n"
  "      unsigned int size_i, \n"
  "      unsigned int size_j, \n"
  "      unsigned int src_rows, \n"
  "      unsigned int src_cols, \n"
  "      unsigned int dst_rows, \n"
  "      unsigned int dst_cols) { \n"
  "  int i = get_global_id(0); \n"
  "  int j = get_global_id(1); \n"
  "  if ( (i+src_offset_i) < src_rows && (j+src_offset_j) < src_cols \n"
  "    && (i+dst_offset_i) < dst_rows && (j+dst_offset_j) < dst_cols ) { \n"
  "    dst((dst_offset_i+i), (dst_offset_j+j)) = \n"
  "      src((src_offset_i+i),(src_offset_j+j)); \n"
  "  } \n"
  "}\n";

  }
}
#endif
