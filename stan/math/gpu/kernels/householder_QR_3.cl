R"(
/*
Calculates res = Ml * Vc, where Ml is left part of matrix M, and Vc is a column of matrix V.

local size: 64
*/
__kernel void householder_QR_3(const int total_rows, const int total_cols, const int ncols,
							  const __global double* M, const __global double* V, __global double* res) {
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lsize = get_local_size(0);
    const int ngroups = get_num_groups(0);
	const int wgid = get_group_id(0);
	__local double v_loc[64];
	double acc = 0;
	
	M += total_rows * gid;
	V += lid + ncols*total_rows;
	for (int k = 0; k < total_rows; k += 64) {
		int end = min(64,total_rows-k);
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
	
	if(gid < ncols){
		res[gid]=acc;
	}
}
;)"