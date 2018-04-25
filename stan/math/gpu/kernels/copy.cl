R"=====(

#define A(i,j)  A[j*rows+i]
#define AT(i,j)  A[j*cols+i]
#define B(i,j)  B[j*rows+i]
#define BT(i,j)  B[j*cols+i]
#define C(i,j)  C[j*rows+i]

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double not supported by OpenCL implementation."
#endif

__kernel void copy(
          __global double *A,
          __global double *B,
          unsigned int rows,
          unsigned int cols) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if ( i < rows && j < cols ) {
     B(i,j) = A(i,j);
    }
}

)====="