CC=gcc
MPICC=mpicc
CFLAGS=-Wall -O3
LDFLAGS=-lm

all: seq omp mpi hybrid

seq: convolution.c
	$(CC) $(CFLAGS) -o convolution convolution.c $(LDFLAGS)

omp: convolution_omp.c
	$(CC) $(CFLAGS) -fopenmp -o conv_omp convolution_omp.c $(LDFLAGS)

mpi: convolution_mpi.c
	$(MPICC) $(CFLAGS) -o conv_mpi convolution_mpi.c $(LDFLAGS)

hybrid: convolution_hybrid.c
	$(MPICC) $(CFLAGS) -fopenmp -o conv_hybrid convolution_hybrid.c $(LDFLAGS)

clean:
	rm -f convolution conv_omp conv_mpi conv_hybrid
