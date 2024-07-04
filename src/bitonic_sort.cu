#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <fstream>

#define MAX_THREADS_PER_BLOCK 1024

void bitonicSortCPU(int* arr, int n) 
{
    for (int k = 2; k <= n; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < n; ++i) {
                int ij = i ^ j;

                if (ij > i) {
                    if ((i & k) == 0) {
                        if (arr[i] > arr[ij]){
                            std::swap(arr[i], arr[ij]);
                        }
                    } else {
                        if (arr[i] < arr[ij]){
                            std::swap(arr[i], arr[ij]);
                        }
                    }
                }
            }
        }
    }
}

__global__ void bitonicSortGPU(int* arr, int j, int k)
{
    unsigned int i, ij;

    i = threadIdx.x + blockDim.x * blockIdx.x;

    ij = i ^ j;

    if (ij > i) {
        if ((i & k) == 0) {
            if (arr[i] > arr[ij]) {
                std::swap(arr[i], arr[ij]);
            }
        } else {
            if (arr[i] < arr[ij]) {
                std::swap(arr[i], arr[ij]);
            }
        }
    }
}

void printArray(int* arr, int size) 
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

bool isSorted(int* arr, int size) 
{
    for (int i = 1; i < size; ++i) 
    {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

bool isPowerOfTwo(int num) 
{
    return num > 0 && (num & (num - 1)) == 0;
}

int nextPowerOfTwo(int n) 
{
    if (n && !(n & (n - 1))) {
        return n;
    }
    
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;

    return n;
}