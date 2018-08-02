R"(
#ifndef A
#define A(i, j)  A[j * rows + i]
#endif
__kernel void gp_periodic_cov(__global double *x, __global double *cnst, __global double *C, int size) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if ( i < size && j < size && i > j ){
    double dist = x[i]-x[j];
    dist = sqrt(dist*dist);
    double a = cnst[0]*exp(sin(cnst[2] * dist)*sin(cnst[2] * dist)*cnst[1]);
    C[j*size+i] = a;
    C[i*size+j] = a;
    
  }else if ( i < size && j < size && i == j ){
    C[j*size+j] = cnst[0];
  }
};)"
