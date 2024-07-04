CXX = nvcc
NUM_DATA = 512
INPUT = generated_data.txt
OUTPUT = output.txt

all: clean build run

build:
	$(CXX) src/bitonic_sort.cu -o bin/bitonic_sort.out
	$(CXX) src/data_populator.cu -o bin/data_populator.out

run:
	bin/data_populator.out $(ARGS) --input data/$(INPUT) 
	bin/bitonic_sort.out > data/$(OUTPUT)

clean:
	rm -f bin/*
	rm -f data/*