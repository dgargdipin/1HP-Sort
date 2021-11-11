#pragma once
#include "cuda_runtime.h"
#include<vector>
/**
* Function:printArray
*   prints an array stored on host
*   @param a:pointer to input array
*   @param n: length of input array a
*   @return: null
*/
void printArray(int* a, int n);
/**
* Function:debugArray
*   prints an array stored on device along with debugging string
*   @param a:character string to be printed for debugging
*   @param arr:pointer to input array
*   @param n: length of input array a
*   @return: null
*/
void debugArray(char a[], int* arr, int n);

/**
* Function: prefix_sum_on_gpu
*   performs a prefix sum on input array stored in gpu and stores it in output
*   @param data: pointer to input array(should be malloced on device)
*   @param output: pointer to output array(should be malloced on device)
*   @param size: length of data
*/

void prefix_sum_on_gpu(int* data, int* output, int size);

/**
 * Function:  createHistogram
 *
 * generates histogram of input array a of size n and stores it in h
 *
 *
 *  @param a: input array of size n
 *  @param h: output array pointer(should have sufficient space)
 *  @param N: size of array a
 *  @return: null
 */
__global__ void createHistogram(int* a, int* h, int N);

/**
 * Function:  one_hp_sort
 *
 * sorts an array using 1-HP algorithm
 *
 *
 *  @param x: input array of size n stored on device
 *  @param N: size of array x
 *  @param minVal: minimum integer in x
 *  @param maxVal: maximum integer in x + 1
 *  @return: pointer to sorted array stored on device
 */
int* one_hp_sort(int* x, int N, int minVal, int maxVal);

/**
* Function : OneHpTail
* performs the last step in 1-HP algorithm
* @param minVal: minVal of input array
* @param maxVal: maxVal of input array
* @param Ap: pointer to Ap(prefix sum of (A)histogram of x) stored on device
* @param y: pointer to output array stored on device
*/

__global__ void OneHpTail(int minVal, int maxVal, int* Ap, int* y);


/**
* Function: test_one_hp
* Performs hp sort on input array, and outputs the result along with time taken
*	@param d_x:input array stored on device
*	@param N: length of input array d_x
*	@param minVal: minVal of d_x
*	@param maxVal: maxVal of d_x
*	@param verbose: if verbose is on input and output array will be printed
*/
void test_one_hp(int* d_x, int N, int minVal, int maxVal, bool verbose);


/**
 * Function:  cubsort
 *
 * sorts an array using inbuilt parallelised radixsort algorithm
 *
 *
 *  @param x: input array of size n stored on device
 *  @param N: size of array x
 *  @param minVal: minimum integer in x
 *  @param maxVal: maximum integer in x + 1
 *  @return: pointer to sorted array stored on device
 */
int* cubsort(int* d_x, int N, int minVal, int maxVal);

/**
* Function: test_cubsort
* Performs cubsort on input array, and outputs the result along with time taken
*	@param d_x:input array stored on device
*	@param N: length of input array d_x
*	@param minVal: minVal of d_x
*	@param maxVal: maxVal of d_x
*	@param verbose: if verbose is on input and output array will be printed
*/
void test_cubsort(int* d_x, int N, int minVal, int maxVal, bool verbose);

/**
* Function: test_sort
* Wrapper to test_cubsort and test_one_hp which takes an additional argument of function address
* which is supposed to sort the input array. It prints the input and output(sorted) arrays along with
* the time taken to sort the input by the argument function.
*	@param d_x:input array stored on device
*	@param N: length of input array d_x
*	@param minVal: minVal of d_x
*	@param maxVal: maxVal of d_x
*	@param func: sort function argument
*	@param verbose: if verbose is on input and output array will be printed
*/
void test_sort(int* d_x, int N, int minVal, int maxVal, int* (*func)(int*, int, int, int), bool verbose);


/**
* Function:generate_random_unique_array
* Generates a random vector of size N, with maximum range of values M
* @param N:size of generated array
* @param M: range of generated array
* @return : random vector of size N, with maximum range of values M
*/
std::vector<int> generate_random_unique_array(int N, int M);

void getInput(int& N, int& M, bool& verbose);
