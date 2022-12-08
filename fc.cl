__kernel void fc(__global float* input_neuron,
	__global float* output_neuron,
	__global float* weights,
	__global float* biases,
	__local float* local_sum,
	int M,
	int N) {

	int j = get_global_id(0);
	int i = get_local_id(1);
	int input_index_i = get_global_id(1);
	int local_id = get_local_id(1);
	int image_num = get_global_id(1) / get_local_size(1);

	local_sum[i] = input_neuron[input_index_i] * weights[j * N + i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int p = N / 2; p >= 1; p = p >> 1) {
		if (local_id < p) local_sum[local_id] += local_sum[local_id + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_id == 0) {
		local_sum[0] += biases[j];

		output_neuron[j + M * image_num] = (local_sum[0] > 0) ? local_sum[0] : 0;
	}


}



