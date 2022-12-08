__kernel void convolution(__global float* inputs,
	__global float* outputs,
	__local float* local_sum,
	__global float* filters,
	__global float* biases,
	int N,
	int D2) 
{
	int output_channel_num = get_global_id(1); // D2 => j  0~63
	int input_channel_num = get_local_id(0); // D1 => i 0~2
	int input_channel_size = get_local_size(0); // D1 == 3  
	int local_id = get_local_id(0); // 0~2
	int matrix_index_num = get_global_id(2); // N * i + j
	int image_num = get_global_id(0) / get_local_size(0); // 0 ~ 2999 / 3

	float* input = inputs + N * N * get_global_id(0);
	float* output = outputs + N * N * get_global_id(1) + N * N * D2 * image_num;
	float* filter = filters + 3 * 3 * (output_channel_num * input_channel_size + input_channel_num);
	float bias = biases[output_channel_num];
	float sum = 0.0;

	int i = matrix_index_num / N;
	int j = matrix_index_num % N;

	/* convolution 3x3 */
	for (int k = 0; k < 3; k++) {
		for (int l = 0; l < 3; l++) {
			int x = i + k - 1;
			int y = j + l - 1;
			if (x >= 0 && x < N && y >= 0 && y < N)
				sum += input[x * N + y] * filter[k * 3 + l];
		}
	}

	local_sum[local_id] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);



	if (input_channel_size % 2 != 0) {
		if (local_id == 0) {
			output[i * N + j] = (((local_sum[0] + local_sum[1] + local_sum[2]) > 0) ? (local_sum[0] + local_sum[1] + local_sum[2]) : 0);
		}
	}
	else {
		for (int p = input_channel_size / 2; p >= 1; p = p >> 1) {
			if (local_id < p) local_sum[local_id] += local_sum[local_id + p];
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if (local_id == 0) {
			output[i * N + j] = (((local_sum[0] + bias) > 0) ? (local_sum[0] + bias) : 0);
			/*if (i == 0 && j == 0 && local_id == 0 && output_channel_num == 0) {
				printf("write %f in output[%d]\n", local_sum[0], i * N + j);
			}*/
		}
	}
}





