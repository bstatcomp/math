R"(
/*
Calculates Mc = Vc - Ml * v, where Mc is a column of the matrix M, Vc is a column of the matrix V, Ml is left part of the matrix M and v is a vector.

local size: 64
*/
__kernel void householder_QR_4(const int total_rows, const int total_cols, const int ncols,
							  __global double* M, const __global double* v, const __global double* V) {
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lsize = get_local_size(0);
    const int ngroups = get_num_groups(0);
	const int wgid = get_group_id(0);
	__local double v_loc[64];
	double acc = 0;
	
	M += gid;
	__global double* M_start=M;
	v += lid;
	for (int k = 0; k < ncols; k += 64) {
		int end = min(64,ncols-k);
		if(lid<end){
			v_loc[lid] = *v;
			v += 64;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if(gid < total_rows){
			for (int j = 0; j < end; j++) {
				acc += *M * v_loc[j];
				M += total_rows;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(gid<total_rows){
		M_start[total_rows*ncols] = V[total_rows*ncols+gid] - acc;
	}
}
;)"