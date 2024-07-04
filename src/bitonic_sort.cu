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