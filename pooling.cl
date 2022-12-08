__kernel void pooling(__global float* inputs,
	__global float* outputs,
	int N) {

	int i = get_local_id(0); // 0 ~ D-1
	int matrix_index_num = get_global_id(1); // N * i + j
	int image_num = get_global_id(0) / get_local_size(0); // 0 ~ D-1 * 1000 / D


	int col_num = matrix_index_num / N;
	int row_num = matrix_index_num % N;

	float* input = inputs + get_global_id(0) * N * N * 4;
	float* output = outputs + i * N * N + N * N * get_local_size(0) * image_num;

	float max = 0.0;

	/* pooling2x2 */
	for (int k = 0; k < 2; k++) {
		for (int l = 0; l < 2; l++) {
			float pixel = input[(col_num * 2 + k) * 2 * N + row_num * 2 + l];
			max = (max > pixel) ? max : pixel;
		}
	}

	output[col_num * N + row_num] = max;


}





