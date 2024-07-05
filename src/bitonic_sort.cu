#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <fstream>

#define MAX_THREADS_PER_BLOCK 1024

const char* default_input_filename = "data/generated_data.txt";
const char* default_output_filename = "data/output.txt";

/*
The bitonic sort is a favourable algorithm for implementation on parallel systems(i.e. gpu) because elements are 
compared in a predefined sequence. The sequence of the comparisons has no effect on the result of the algorithm.  

Taking an example to explain the algorithm implemented here,
arr=[3,7,4,8,6,2,1,5]

k=2,j=1,i=0,i^j=1: no swap
k=2,j=1,i=1,i^j=0: no swap
k=2,j=1,i=2,i^j=3: swapped! arr=[3,7,8,4,6,2,1,5]
k=2,j=1,i=3,i^j=2: no swap
k=2,j=1,i=4,i^j=5: swapped! arr=[3,7,8,4,2,6,1,5]
k=2,j=1,i=5,i^j=4: no swap
k=2,j=1,i=6,i^j=7: swapped! arr=[3,7,8,4,2,6,5,1]
k=2,j=1,i=7,i^j=6: no swap

k=4,j=2,i=0,i^j=2: no swap
k=4,j=2,i=1,i^j=3: swapped! arr=[3,4,8,7,2,6,5,1]
k=4,j=2,i=2,i^j=0: no swap
k=4,j=2,i=3,i^j=1: no swap
k=4,j=2,i=4,i^j=6: swapped! arr=[3,4,8,7,5,6,2,1]
k=4,j=2,i=5,i^j=7: no swap
k=4,j=2,i=6,i^j=4: no swap
k=4,j=2,i=7,i^j=5: no swap
k=4,j=1,i=0,i^j=1: no swap
k=4,j=1,i=1,i^j=0: no swap
k=4,j=1,i=2,i^j=3: swapped! arr=[3,4,7,8,5,6,2,1]
k=4,j=1,i=3,i^j=2: no swap
k=4,j=1,i=4,i^j=5: swapped! arr=[3,4,7,8,6,5,2,1]
k=4,j=1,i=5,i^j=4: no swap
k=4,j=1,i=6,i^j=7: no swap
k=4,j=1,i=7,i^j=6: no swap

k=8,j=4,i=0,i^j=4: no swap
k=8,j=4,i=1,i^j=5: no swap
k=8,j=4,i=2,i^j=6: swapped! arr=[3,4,2,8,6,5,7,1]
k=8,j=4,i=3,i^j=7: swapped! arr=[3,4,2,1,6,5,7,8]
k=8,j=4,i=4,i^j=0: no swap
k=8,j=4,i=5,i^j=1: no swap
k=8,j=4,i=6,i^j=2: no swap
k=8,j=4,i=7,i^j=3: no swap
k=8,j=2,i=0,i^j=2: swapped! arr=[2,4,3,1,6,5,7,8]
k=8,j=2,i=1,i^j=3: swapped! arr=[2,1,3,4,6,5,7,8]
k=8,j=2,i=2,i^j=0: no swap
k=8,j=2,i=3,i^j=1: no swap
k=8,j=2,i=4,i^j=6: no swap
k=8,j=2,i=5,i^j=7: no swap
k=8,j=2,i=6,i^j=4: no swap
k=8,j=2,i=7,i^j=5: no swap
k=8,j=1,i=0,i^j=1: swapped! arr=[1,2,3,4,6,5,7,8]
k=8,j=1,i=1,i^j=0: no swap
k=8,j=1,i=2,i^j=3: no swap
k=8,j=1,i=3,i^j=2: no swap
k=8,j=1,i=4,i^j=5: swapped! arr=[1,2,3,4,5,6,7,8]
k=8,j=1,i=5,i^j=4: no swap
k=8,j=1,i=6,i^j=7: no swap
k=8,j=1,i=7,i^j=6: no swap

*/

//implementation of the bitonic sort for the cpu
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

//implementation of the bitonic sort for the gpu
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

//sort checker
bool isSorted(int* arr, int size) 
{
    for (int i = 1; i < size; ++i) 
    {
        if (arr[i] < arr[i - 1])
            return false;
    }
    return true;
}

//is the integer provided a power of two
//isPowerOfTwo(x) = true , if there exists an integer k such that x = 2^k
//                = false, otherwise
bool isPowerOfTwo(int num) 
{
    return num > 0 && (num & (num - 1)) == 0;
}

//finds the next largest power of two for a given 16-bit integer
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

int main(int argc, char* argv[]) 
{   
    const char* input_filename = default_input_filename;
    const char* output_filename = default_output_filename;

    //get input and output filenames through argv[] via flags
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_filename = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_filename = argv[++i];
        } else {
            std::cerr << "usage: " << argv[0] << " [--input filename] [--output filename]\n";
            return 1;
        }
    }

    std::ifstream infile(input_filename);
    if (!infile) {
        std::cerr << "error opening file: " << input_filename << "\n";
        return 1;
    }

    //first number in the file is the count of numbers
    int input_size = 0;
    infile >> input_size;
    int size;

    if (input_size <= 0) {
        std::cerr << "array size must be a positive integer\n";
        return 1;
    }

    if (!isPowerOfTwo(input_size)) {   
        std::cout << 
        "size provided is not a power of two, size will be the next power of two and  
        remaining spots of the array will be padded with zeroes\nsize provided: " 
        << input_size << std::endl;
        size = nextPowerOfTwo(input_size);
        std::cout << "the nearest higher power of two is: " << size << std::endl;
    } else {
        size = input_size;
    }

    int* arr = new int[size];
    int* cpu_arr = new int[size];

    int* gpu_arr;

    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < input_size; ++i) {
        if (!(infile >> arr[i])) {
            std::cerr << "error reading number at position " << i + 1 << ".\n";
            delete[] arr;
            return 1;
        }
        cpu_arr[i] = arr[i];
    }

    infile.close();

    //if input_size < size (in the case that input_size is not a power of two) we populate the remaining empty slots of the array with 0
    for (int i = input_size; i < size; ++i) {
        arr[i] = 0;
        cpu_arr[i] = 0;
    }

    //allocate memory on gpu 
    cudaMalloc((void**)&gpu_arr, size * sizeof(int));

    //copy arr[] onto the gpu in the form of gpu_arr[]
    cudaMemcpy(gpu_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    float GPU_time_ms = 0;

    clock_t startCPU, endCPU;

    //threads per block and blocks per grid used for spawning the kernel 
    int threadsPerBlock = MAX_THREADS_PER_BLOCK;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int j, k;

    cudaEventRecord(startGPU);
    for (k = 2; k <= size; k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            //using the gpu for only the innermost loop
            bitonicSortGPU <<<blocksPerGrid, threadsPerBlock>>> (gpu_arr, j, k);
        }
    }
    cudaEventRecord(stopGPU);

    //load the sorted contents from gpu_arr[] into arr[] 
    cudaMemcpy(arr, gpu_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stopGPU); //wait for all gpu threads to finish their work
    cudaEventElapsedTime(&GPU_time_ms, startGPU, stopGPU);

    startCPU = clock();
    bitonicSortCPU(cpu_arr, size);
    endCPU = clock();

    double CPU_time_ms = static_cast<double>(endCPU - startCPU) / (CLOCKS_PER_SEC / 1000.0);
    
    if (isSorted(arr, size))
        std::cout << "\n\nsort checker: gpu array success" << std::endl;
    else
        std::cout << "sort checker: gpu array fail" << std::endl;
   
    if (isSorted(cpu_arr, size))
        std::cout << "sort checker: cpu array success" << std::endl;
    else
        std::cout << "sort checker: cpu array fail" << std::endl;

    std::cout << "\n\ngpu time: " << GPU_time_ms << " ms" << std::endl;
    std::cout << "cpu time: " << CPU_time_ms << " ms" << std::endl;

    std::ofstream outfile(output_filename);
    if (!outfile) {
        std::cerr << "error opening output file.\n";
        delete[] arr;
        return 1;
    }

    for (int i = size-input_size; i < size; ++i) {
        outfile << arr[i] << " ";
    }

    outfile.close();

    //free all the arrays created
    delete[] arr;
    delete[] cpu_arr;
    cudaFree(gpu_arr);

    return 0;
}
