R"(
__kernel void cov_exp_quad2(__global double *x, __global double *cnst, __global double *res, int size) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if ( i < size && j < (size-1) && i >= (j+1) ){
    double d = x[i]-x[j];    
    double a = cnst[0]*exp(cnst[1]*(d*d));
    res[j*size+i] = a;
    res[i*size+j] = a;
  }else if( i == j ){
    res[j*size+i] = cnst[0];
  }
};)"
