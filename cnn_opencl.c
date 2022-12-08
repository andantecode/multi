#define _CRT_SECURE_NO_WARNINGS
#include<CL/cl.h>
#include"cnn.h"
#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<math.h>

#define ReLU(x) (((x)>0)?(x):0)
#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
		exit(EXIT_FAILURE); \
	}
#define parallel_size 10

char* get_source_code(const char* file_name, size_t* len) {
	char* source_code;
	char buf[2] = "\0";
	int cnt = 0;
	size_t length;
	FILE* file = fopen(file_name, "r");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}
	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);
	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	for (int i = 0; i < length; i++) {
		buf[0] = source_code[i];
		if (buf[0] == '\n') cnt++;
	}
	source_code[length - cnt] = '\0';
	fclose(file);
	*len = length - cnt;
	return source_code;
}

static int platformNum = 0;
static int deviceNum = 0;


static cl_uint platformCount;
static cl_platform_id* platforms;
static cl_uint deviceCount;
static cl_device_id* devices;
static cl_device_id device;
static cl_context context;
static cl_int err;
static cl_command_queue queue;

size_t convolution_kernel_source_size;
size_t pooling_kernel_source_size;
size_t fc_kernel_source_size;

char* convolution_kernel_source;
char* pooling_kernel_source;
char* fc_kernel_source;

char* sourceFile;
char* kernelName;

static cl_program convolution_program;
static cl_program pooling_program;
static cl_program fc_program;

cl_int convolution_build_status;
cl_int pooling_build_status;
cl_int fc_build_status;

static cl_kernel convolution_kernel;
static cl_kernel pooling_kernel;
static cl_kernel fc_kernel;

cl_mem inputsBuffer;
cl_mem filtersBuffer;
cl_mem outputsBuffer;
cl_mem biasesBuffer;
cl_mem weightsBuffer;

void cnn_init() {
	// 1. platform
	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	// 2. device
	clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
	clGetDeviceIDs(platforms[platformNum], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
	device = devices[deviceNum];

	// 3. context
	context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

	// 4. command queue
	queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);

	/* convolution initialize */

	sourceFile = "convolution.cl";
	kernelName = "convolution";

	// 5. get source
	convolution_kernel_source = get_source_code(sourceFile, &convolution_kernel_source_size);

	// 6. program build
	convolution_program = clCreateProgramWithSource(context, 1, (const char**)&convolution_kernel_source, &convolution_kernel_source_size, NULL);
	convolution_build_status = clBuildProgram(convolution_program, 1, &device, NULL, NULL, NULL);

	// 7. create kernel
	convolution_kernel = clCreateKernel(convolution_program, kernelName, &err);

	// 8. create memory buffer
	inputsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, parallel_size * 512 * 32 * 32 * sizeof(float), NULL, NULL);
	filtersBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * 3 * 3 * sizeof(float), NULL, NULL);
	biasesBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * sizeof(float), NULL, NULL);
	weightsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, 512 * 512 * sizeof(float), NULL, NULL);

	/* pooling initialize */

	sourceFile = "pooling.cl";
	kernelName = "pooling";

	// 5. get source
	pooling_kernel_source = get_source_code(sourceFile, &pooling_kernel_source_size);

	// 6. program build
	pooling_program = clCreateProgramWithSource(context, 1, (const char**)&pooling_kernel_source, &pooling_kernel_source_size, NULL);
	pooling_build_status = clBuildProgram(pooling_program, 1, &device, NULL, NULL, NULL);

	// 7. create kernel
	pooling_kernel = clCreateKernel(pooling_program, kernelName, &err);

	/* final connect initialize */

	sourceFile = "fc.cl";
	kernelName = "fc";

	// 5. get source 
	fc_kernel_source = get_source_code(sourceFile, &fc_kernel_source_size);

	// 6. program build
	fc_program = clCreateProgramWithSource(context, 1, (const char**)&fc_kernel_source, &fc_kernel_source_size, NULL);
	fc_build_status = clBuildProgram(fc_program, 1, &device, NULL, NULL, NULL);

	// 7. create kernel
	fc_kernel = clCreateKernel(fc_program, kernelName, &err);


}

float* alloc_layer(size_t n) {
	return (float*)malloc(parallel_size * n * sizeof(float));
}


/** convolution_layer
	* D2 = output channel size
	* D1 = input channel size
	* N = width and height of an input image
	* input image is zero-padded by 1.
	* Thus, input is (D1, N, N) and output is (D2, N, N)
	**/
static void convolution_layer(float* inputs, float* outputs, float* filters, float* biases, int D2, int D1, int N) {

	outputsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, parallel_size * D2 * N * N * sizeof(float), NULL, NULL);

	// 9. Insert Memory Buffer
	clEnqueueWriteBuffer(queue, inputsBuffer, CL_TRUE, 0, D1 * N * N * parallel_size * sizeof(float), inputs, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, filtersBuffer, CL_TRUE, 0, D1 * D2 * 3 * 3 * sizeof(float), filters, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, biasesBuffer, CL_TRUE, 0, D2 * sizeof(float), biases, 0, NULL, NULL);

	// 10. Set Kernel Argument
	clSetKernelArg(convolution_kernel, 0, sizeof(cl_mem), &inputsBuffer);
	clSetKernelArg(convolution_kernel, 1, sizeof(cl_mem), &outputsBuffer);
	clSetKernelArg(convolution_kernel, 2, sizeof(float) * 512, NULL);
	clSetKernelArg(convolution_kernel, 3, sizeof(cl_mem), &filtersBuffer);
	clSetKernelArg(convolution_kernel, 4, sizeof(cl_mem), &biasesBuffer);
	clSetKernelArg(convolution_kernel, 5, sizeof(int), &N);
	clSetKernelArg(convolution_kernel, 6, sizeof(int), &D2);

	// 11. Insert Kernel
	size_t globalSize[3] = { D1 * parallel_size, D2, N * N };
	size_t localSize[3] = { D1, 1, 1 };

	clEnqueueNDRangeKernel(queue, convolution_kernel, 3, NULL, globalSize, localSize, 0, NULL, NULL);

	// 12. return outputBuffer
	clEnqueueReadBuffer(queue, outputsBuffer, CL_TRUE, 0, parallel_size * D2 * N * N * sizeof(float), outputs, 0, NULL, NULL);

	// 13. wait
	clFlush(queue);
	clFinish(queue);

	/*printf("=====================================\n");
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf(" %f", outputs[i * N + j]);
		}
		printf("\n");
	}

	system("pause");*/

	clReleaseMemObject(outputsBuffer);
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
static void pooling_layer(float* inputs, float* outputs, int D, int N) {
	outputsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, parallel_size * D * N * N * sizeof(float), NULL, NULL);

	// 9. Insert Memory Buffer
	clEnqueueWriteBuffer(queue, inputsBuffer, CL_TRUE, 0, parallel_size * D * N * 2 * N * 2 * sizeof(float), inputs, 0, NULL, NULL);

	// 10. Set Kernel Argument
	clSetKernelArg(pooling_kernel, 0, sizeof(cl_mem), &inputsBuffer);
	clSetKernelArg(pooling_kernel, 1, sizeof(cl_mem), &outputsBuffer);
	clSetKernelArg(pooling_kernel, 2, sizeof(int), &N);

	// 11. Insert Kernel
	size_t globalSize[2] = { D * parallel_size, N * N };
	size_t localSize[2] = { D, 1 };
	clEnqueueNDRangeKernel(queue, pooling_kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);

	// 12. return outputBuffer
	clEnqueueReadBuffer(queue, outputsBuffer, CL_TRUE, 0, parallel_size * D * N * N * sizeof(float), outputs, 0, NULL, NULL);

	// 13. wait
	clFlush(queue);
	clFinish(queue);

	/*for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf(" %f", outputs[N * N * 65 + i * N + j]);
		}
		printf("\n");
	}

	printf("==========================================\n");

	system("pause");*/

	clReleaseMemObject(outputsBuffer);
}

/*
 * M = output size
 * N = input size
 */
static void fc_layer(float* input_neuron, float* output_neuron, float* weights, float* biases, int M, int N) {
	outputsBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, parallel_size * M * sizeof(float), NULL, NULL);

	// 9. Insert Memory Buffer
	clEnqueueWriteBuffer(queue, inputsBuffer, CL_TRUE, 0, parallel_size * N * sizeof(float), input_neuron, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, weightsBuffer, CL_TRUE, 0, N * M * sizeof(float), weights, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, biasesBuffer, CL_TRUE, 0, M * sizeof(float), biases, 0, NULL, NULL);

	// 10. Set Kernel Argument
	clSetKernelArg(fc_kernel, 0, sizeof(cl_mem), &inputsBuffer);
	clSetKernelArg(fc_kernel, 1, sizeof(cl_mem), &outputsBuffer);
	clSetKernelArg(fc_kernel, 2, sizeof(cl_mem), &weightsBuffer);
	clSetKernelArg(fc_kernel, 3, sizeof(cl_mem), &biasesBuffer);
	clSetKernelArg(fc_kernel, 4, sizeof(float) * N, NULL);
	clSetKernelArg(fc_kernel, 5, sizeof(int), &M);
	clSetKernelArg(fc_kernel, 6, sizeof(int), &N);

	// 11. Insert Kernel
	size_t globalSize[2] = { M, N * parallel_size };
	size_t localSize[2] = { 1, N };
	clEnqueueNDRangeKernel(queue, fc_kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);

	// 12. return outputBuffer
	clEnqueueReadBuffer(queue, outputsBuffer, CL_TRUE, 0, parallel_size * M * sizeof(float), output_neuron, 0, NULL, NULL);

	// 13. wait
	clFlush(queue);
	clFinish(queue);

	clReleaseMemObject(outputsBuffer);
}

static void softmax(float* output, int N) {
	int i;
	float max = output[0];
	for (i = 1; i < N; i++) {
		max = (output[i] > max) ? output[i] : max;
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(output[i] - max);
	}
	for (i = 0; i < N; i++) {
		output[i] = exp(output[i] - max) / sum;
	}
}

static int find_max(float* fc, int N) {
	int i;
	int maxid = 0;
	float maxval = 0;
	for (i = 0; i < N; i++) {
		if (maxval < fc[i]) {
			maxval = fc[i];
			maxid = i;
		}
	}
	return maxid;
}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {

	float* w1_1, * b1_1, * w1_2, * b1_2;
	float* w2_1, * b2_1, * w2_2, * b2_2;
	float* w3_1, * b3_1, * w3_2, * b3_2, * w3_3, * b3_3;
	float* w4_1, * b4_1, * w4_2, * b4_2, * w4_3, * b4_3;
	float* w5_1, * b5_1, * w5_2, * b5_2, * w5_3, * b5_3;
	float* w1, * b1, * w2, * b2, * w3, * b3;
	w1_1 = network[0]; b1_1 = network[1];
	w1_2 = network[2]; b1_2 = network[3];
	w2_1 = network[4]; b2_1 = network[5];
	w2_2 = network[6]; b2_2 = network[7];
	w3_1 = network[8]; b3_1 = network[9];
	w3_2 = network[10]; b3_2 = network[11];
	w3_3 = network[12]; b3_3 = network[13];
	w4_1 = network[14]; b4_1 = network[15];
	w4_2 = network[16]; b4_2 = network[17];
	w4_3 = network[18]; b4_3 = network[19];
	w5_1 = network[20]; b5_1 = network[21];
	w5_2 = network[22]; b5_2 = network[23];
	w5_3 = network[24]; b5_3 = network[25];
	w1 = network[26]; b1 = network[27];
	w2 = network[28]; b2 = network[29];
	w3 = network[30]; b3 = network[31];

	float* c1_1, * c1_2, * p1;
	float* c2_1, * c2_2, * p2;
	float* c3_1, * c3_2, * c3_3, * p3;
	float* c4_1, * c4_2, * c4_3, * p4;
	float* c5_1, * c5_2, * c5_3, * p5;
	float* fc1, * fc2, * fc3;

	c1_1 = alloc_layer(64 * 32 * 32);
	c1_2 = alloc_layer(64 * 32 * 32);
	p1 = alloc_layer(64 * 16 * 16);
	c2_1 = alloc_layer(128 * 16 * 16);
	c2_2 = alloc_layer(128 * 16 * 16);
	p2 = alloc_layer(128 * 8 * 8);
	c3_1 = alloc_layer(256 * 8 * 8);
	c3_2 = alloc_layer(256 * 8 * 8);
	c3_3 = alloc_layer(256 * 8 * 8);
	p3 = alloc_layer(256 * 4 * 4);
	c4_1 = alloc_layer(512 * 4 * 4);
	c4_2 = alloc_layer(512 * 4 * 4);
	c4_3 = alloc_layer(512 * 4 * 4);
	p4 = alloc_layer(512 * 2 * 2);
	c5_1 = alloc_layer(512 * 2 * 2);
	c5_2 = alloc_layer(512 * 2 * 2);
	c5_3 = alloc_layer(512 * 2 * 2);
	p5 = alloc_layer(512 * 1 * 1);
	fc1 = alloc_layer(512);
	fc2 = alloc_layer(512);
	fc3 = alloc_layer(10);

	for (int i = 0; i < num_images / parallel_size; ++i)
	{
		float* image = images + parallel_size * i * 3 * 32 * 32;

		

		convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32);
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
		pooling_layer(c1_2, p1, 64, 16);

		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16);
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16);
		pooling_layer(c2_2, p2, 128, 8);

		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8);
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8);
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8);
		pooling_layer(c3_3, p3, 256, 4);

		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4);
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4);
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4);
		pooling_layer(c4_3, p4, 512, 2);

		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2);
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2);
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2);
		pooling_layer(c5_3, p5, 512, 1);

		fc_layer(p5, fc1, w1, b1, 512, 512);
		fc_layer(fc1, fc2, w2, b2, 512, 512);
		fc_layer(fc2, fc3, w3, b3, 10, 512);

		/*if (i == 1) {
			for (int j = 0; j < 10; j++) {
				printf(" %f\n", fc3[j]);
			}
		}*/



		for (int j = 0; j < parallel_size; j++) {
			float* point = fc3 + j * 10;

			softmax(point, 10);
			labels[i * parallel_size + j] = find_max(point, 10);
			confidences[i * parallel_size + j] = point[labels[i * parallel_size + j]];
		}


	}

	free(c1_1); free(c1_2); free(p1);
	free(c2_1); free(c2_2); free(p2);
	free(c3_1); free(c3_2); free(c3_3); free(p3);
	free(c4_1); free(c4_2); free(c4_3); free(p4);
	free(c5_1); free(c5_2); free(c5_3); free(p5);
	free(fc1); free(fc2); free(fc3);

	/*clReleaseKernel(kernel);
	clReleaseProgram(program);
	free(kernel_source);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseDevice(device);*/

}