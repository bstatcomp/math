R"(
#ifndef A
#define A(i, j)  A[j * rows + i]
#endif
__kernel void gp_dot_prod_cov(__global double *x, __global double *cnst, __global double *C, int size) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  for (size_t i = 0; i < (x_size - 1); ++i) {
    cov(i, i) = sigma_sq + dot_self(x[i]);
    for (size_t j = i + 1; j < x_size; ++j) {
      cov(i, j) = sigma_sq + dot_product(x[i], x[j]);
      cov(j, i) = cov(i, j);
    }
  }
  if ( i < (size-1) && j < size && i > j ){
    double a = cnst[0]*x[i]*x[j];
    C[j*size+i] = a;
    C[i*size+j] = a;    
  }else if ( i < size && j < size && i == j ){
    C[j*size+j] = cnst[0]+x[i]*x[i];
  }
};)"
