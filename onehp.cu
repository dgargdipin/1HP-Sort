#include "cuda_runtime.h"
#include<iostream>
#include "onehp.cuh"
#include<cub/cub.cuh>
#include<algorithm>

void printArray(int* a, int n) {
	for (int i = 0; i < n; i++) {
		printf("%d ", a[i]);
	}
	printf("\n");
}

void debugArray(char a[], int* arr, int n) {
	printf("DEBUGGING %s\n", a);
	int* host_arr = new int[n];
	cudaMemcpy(host_arr, arr, n * sizeof(int), cudaMemcpyDeviceToHost);
	printArray(host_arr, n);
	delete[] host_arr;
}

__global__ void createHistogram(int* a, int* h, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	int pos = a[tid];
	atomicAdd(&h[pos], 1);

}

int* one_hp_sort(int* x, int N, int minVal, int maxVal)
{
	int* y;//output
	cudaMalloc(&y, sizeof(int) * N);
	int* A, * A_p;
	cudaMalloc(&A, sizeof(int) * maxVal);
	cudaMalloc(&A_p, sizeof(int) * maxVal);
	int numThreads = 256;
	int numBlocks = (N + numThreads - 1) / numThreads;
	createHistogram << <numBlocks, numThreads >> > (x, A, N);
	prefix_sum_on_gpu(A, A_p, maxVal);
	OneHpTail << < (maxVal - minVal + 255) / 256, 256 >> > (minVal, maxVal, A_p, y);
	cudaFree(A);
	cudaFree(A_p);
	return y;


}

void prefix_sum_on_gpu(int* data, int* output, int size) {
	void* d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, data, output, size);
	// Allocate temporary storage for inclusive prefix sum
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run inclusive prefix sum
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, data, output, size);
	//printf("Successfully prefixed sum");
}


__global__ void OneHpTail(int minVal, int maxVal, int* Ap, int* y) {
	int i = minVal + blockIdx.x * blockDim.x + threadIdx.x;
	if (i > maxVal)return;
	if (i == minVal) {
		if (Ap[minVal])y[0] = minVal;
	}
	else if (Ap[i - 1] != Ap[i])y[Ap[i - 1]] = i;
}

void test_one_hp(int* d_x, int N, int minVal, int maxVal, bool verbose) {
	printf("-------------Testing 1HP algorithm---------------------\n");
	test_sort(d_x, N, minVal, maxVal, &one_hp_sort, verbose);
};
void test_cubsort(int* d_x, int N, int minVal, int maxVal, bool verbose)
{
	printf("-------------Testing CubSort algorithm---------------------\n");
	test_sort(d_x, N, minVal, maxVal, &cubsort, verbose);
}


int* cubsort(int* d_x, int N, int minVal, int maxVal)
{
	// Declare, allocate, and initialize device-accessible pointers for sorting data
	int  num_items = N;          // e.g., 7
	int* d_keys_in = d_x;         // e.g., [8, 6, 7, 5, 3, 0, 9]
	int* d_keys_out;        // e.g., [        ...        ]
	cudaMalloc(&d_keys_out, sizeof(int) * N);
	// Determine temporary device storage requirements
	void* d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run sorting operation
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items);
	// d_keys_out            <-- [0, 3, 5, 6, 7, 8, 9]
	return d_keys_out;
}




void test_sort(int* d_x, int N, int minVal, int maxVal, int* (*func)(int*, int, int, int), bool verbose) {
	float milliseconds = 0;
	char input_str[] = "input:";
	if (verbose)debugArray(input_str, d_x, N);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	int* d_y = func(d_x, N, minVal, maxVal);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	char output_str[] = "output:";
	if (verbose)debugArray(output_str, d_y, N);
	printf("Took %f milliseconds\n", milliseconds);
	cudaFree(d_y);
}

std::vector<int> generate_random_unique_array(int N, int M) {
	std::vector<int> x(M);


	for (int i = 0; i < M; i++) {
		x[i] = i;
	}
	std::srand(unsigned(std::time(0)));
	std::random_shuffle(x.begin(), x.end());
	x = std::vector<int>(x.begin(), x.begin() + N);
	return x;

};
void getInput(int& N, int& M, bool& verbose) {
	std::cout << "Enter the length of random array: ";
	std::cin >> N;
	std::cout << std::endl;
	std::cout << "Enter the range of random array: ";
	std::cin >> M;
	std::cout << std::endl;
	std::cout << "Do you want to verbose output? (y/n): ";
	char verboseChar;
	std::cin >> verboseChar;
	verbose = (verboseChar == 'y');
}