#include <iostream>
#include "onehp.cuh"
#include<vector>
#include <algorithm>
int main()
{
    constexpr int N = 15000, M = 20000;

    size_t bytesN = sizeof(int) * N;

    std::vector<int> x(N);
    srand(time(0));
    std::generate(x.begin(), x.end(), [M]() {return rand() % M; });

    int* d_x;

    // x is input arr
    // d_x is copy of x on gpu
    cudaMalloc(&d_x, bytesN);
    cudaMemcpy(d_x, x.data(), bytesN, cudaMemcpyHostToDevice);

    test_one_hp(d_x, N,0,M);
    
    test_cubsort(d_x, N,0, M);
    cudaFree(d_x);

    
    
}