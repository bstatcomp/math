R"(
#ifndef A
#define A(i, j)  A[j * rows + i]
#endif
__kernel void cov_exp_quad(__global double *x, __global double *pos, __global double *cnst, __global double *dist, __global double *C, int size) {
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  if ( i < size && j < (size-1) && i >= (j+1) ){
    int p = pos[j*size+i];
    double d = x[i]-x[j];
    dist[p] = d*d;
    C[p] = cnst[0]*exp(cnst[1]*(-dist[p]));
  }
};)"
