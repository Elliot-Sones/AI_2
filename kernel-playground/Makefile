# make test   -> compile with clang on this Mac and run the tests (CPU shim)
# make cuda   -> on a machine with nvcc, compile the SAME files for the real GPU
CC      ?= cc
CFLAGS  ?= -std=c11 -O1 -Wall -Wextra -Wno-unused-function -Wno-unused-parameter
NVCC    ?= nvcc

test: test_kernels
	./test_kernels

test_kernels: test_kernels.c kernels.h cuda_shim.h
	$(CC) $(CFLAGS) -o $@ test_kernels.c -lm

cuda: test_kernels.c kernels.h cuda_shim.h
	$(NVCC) -x cu -O2 -o test_kernels_cuda test_kernels.c
	./test_kernels_cuda

clean:
	rm -f test_kernels test_kernels_cuda

.PHONY: test cuda clean

trace: trace.c cuda_shim.h
	$(CC) $(CFLAGS) -o trace trace.c && ./trace
