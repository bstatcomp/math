R"(
__kernel void cov_exp_quad3(__global double *x1, __global double *x2,__global double *cnst, __global double *res, int size) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if ( i < size && j < size ){
    double d = x1[i]-x2[j];    
    res[j*size+i] = cnst[0]*exp(cnst[1]*(d*d));
  }
};)"
