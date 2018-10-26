R"(
/*
Calculatex R_idx-th householder vector of R and stores it into V_idx-th column of V.

Must be run with 1 work group of 128 threads.
*/
__kernel void householder_QR_2(const int total_rows, const int total_cols, const int R_idx, const int V_idx,
                               const __global double* R, __global double* V){
    const int gid = get_global_id(0);
    const int gsize = get_global_size(0);
	
	__local double acc_loc[128];
	double acc=0;
	for(int i=R_idx+gid;i<total_rows;i+=128){
		double tmp=R[total_rows*R_idx +i];
		acc+=tmp*tmp;
	}
	acc_loc[gid]=acc;
	
	barrier(CLK_LOCAL_MEM_FENCE);
	double first;
	if(gid==0){
		for(int i=1;i<128;i++){
			acc+=acc_loc[i];
		}
		first = R[total_rows*R_idx + R_idx];
		double acc2=acc-first*first;
		first = first - copysign(sqrt(acc),first);
		acc2+=first*first;
		acc_loc[0]=rsqrt(acc2) * sqrt(2.);
		if(R_idx==total_cols-1 || R_idx==total_cols-1){
			acc_loc[0]=1;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	double sqrt2_div_norm=acc_loc[0];
	for(int i=gid+R_idx;i<total_rows;i++){
		double tmp=R[total_rows*R_idx +i]*sqrt2_div_norm;
		if(i==gid+R_idx && gid==0){
			tmp=first*sqrt2_div_norm;
		}
		V[(total_rows - R_idx +V_idx)*V_idx + V_idx + i - R_idx]=tmp;
	}
}
;)"