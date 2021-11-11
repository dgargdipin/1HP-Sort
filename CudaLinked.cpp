#include <iostream>
#include "onehp.cuh"
#include<vector>

int main()
{

	int N, M;
	bool verbose = false;
	getInput(N, M, verbose);

	std::vector<int> x = generate_random_unique_array(N, M);
	int* d_x;

	// x is input arr
	// d_x is copy of x on gpu
	size_t bytesN = sizeof(int) * N;
	cudaMalloc(&d_x, bytesN);
	cudaMemcpy(d_x, x.data(), bytesN, cudaMemcpyHostToDevice);

	test_one_hp(d_x, N, 0, M, verbose);
	test_cubsort(d_x, N, 0, M, verbose);
	cudaFree(d_x);



}