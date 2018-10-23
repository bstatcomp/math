R"(
/*
Calculates res = Mb - v * (Mb^t * v)^t, where Mb is a block of matrix and v is a vector.

M v column-major formatu

local size: 64
*/
__kernel void householder_QR_1(const int total_rows, const int total_cols, const int start_row, const int start_col, const int nrows, const int ncols,
							  __global double* M, const __global double* v, __global double* res) {
    const int lid = get_local_id(0);
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
    const int lsize = get_local_size(0);
    const int ngroups = get_num_groups(0);
	const int wgid = get_group_id(0);
	__local double v_loc[64];
	double acc = 0;
	
	//__global double* M_i=M;
	M += total_rows * (start_col + gid) + start_row;
	//printf("%d start %d (%d %d) (%d %d)\n",gid, total_rows * (start_col + gid) + start_row, start_row, start_col, total_rows, total_cols);
	__global double* M_start=M;
	v += lid;
	const __global double* v_start=v;
	for (int k = 0; k < nrows; k += 64) {
		int end = min(64,nrows-k);
		//printf("gid: %d, lid: %d end: %d, k: %d\n",gid, lid, end, k);
		if(lid<end){
			v_loc[lid] = *v; // 
			v += 64;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if(gid < ncols){
			for (int j = 0; j < end; j++) {
				acc += *M * v_loc[j]; // 
				//printf("acc_tmp[%d]: %lf (+ %lf * %lf)\n",gid, acc, *M, v_loc[j]);
				M += 1;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//res[0]=acc;
	v = v_start;
	M = M_start;
	
	for (int k = 0; k < nrows; k += 64) {
		int end = min(64,nrows-k);
		if(lid<end){
			v_loc[lid] = *v;//*v;
			v += 64;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if(gid < ncols){
			for (int j = 0; j < end; j++) {
				//if(gid * nrows + k + j>=nrows*ncols)printf("%d writing at %d\n",gid,gid * nrows + k + j);
				//if(((M+k+j)-M_i)/8>total_cols*total_rows)printf("%d reading at %d\n",gid,k + j);
				res[gid * nrows + k + j] = M[k + j] - acc * v_loc[j]; //gid * nrows + k + j
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}
;)"