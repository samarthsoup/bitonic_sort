# Instructions for running the program

1. Navigate to the directory that contains the 'run.sh' file.

2. Run <code>./run.sh</code> in the terminal. 

3. Sorted array will be stored in an 'output.txt' file.

# Changing input size using CLI arguments:

1. Go to the 'run.sh' file.

2. Change the variable 'NUM_DATA' to have any value you want(doesn't have to be a power of 2).

Running 'run.sh' without the 'NUM_DATA' will also work(by default the value for 'ARGS' is 512 as mentioned in the Makefile).

3. The variables 'INPUT' and 'OUTPUT' are used to name the input and output files.

They both have default versions too so mentioning them are optional(INPUT = generated_data.txt, OUTPUT = output.txt).

# Bitonic sort using GPU and Cuda

Bitonic sort is a sorting technique that is very easily parallelised. As the size of the input increases, the advantage of using a GPU over a CPU increases.