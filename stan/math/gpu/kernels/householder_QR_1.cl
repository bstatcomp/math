R"(
/*
Calculates res = Mb - v * (Mb^t * v)^t, where Mb is a block of matrix and v is a vector that lies in idx-th column of V.

local size: 64
*/
__kernel void householder_QR_1(const int total_rows, const int total_cols, const int start_row, const int start_col, const int nrows, const int ncols, const int V_rows,
							  __global double* M, const __global double* V, __global double* res) {
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lsize = get_local_size(0);
    const int ngroups = get_num_groups(0);
	const int wgid = get_group_id(0);
	__local double v_loc[64];
	double acc = 0;
	
	M += total_rows * (start_col + gid) + start_row;
	__global double* M_start=M;
	int idx = V_rows - nrows;
	V += lid + idx + idx*V_rows;
	const __global double* V_start=V;
	for (int k = 0; k < nrows; k += 64) {
		int end = min(64,nrows-k);
		if(lid<end){
			v_loc[lid] = *V;
			V += 64;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if(gid < ncols){
			for (int j = 0; j < end; j++) {
				acc += *M * v_loc[j];
				M += 1;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	V = V_start;
	M = M_start;
	
	for (int k = 0; k < nrows; k += 64) {
		int end = min(64,nrows-k);
		if(lid<end){
			v_loc[lid] = *V;
			V += 64;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if(gid < ncols){
			for (int j = 0; j < end; j++) {
				res[gid * nrows + k + j] = M[k + j] - acc * v_loc[j];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
;)"