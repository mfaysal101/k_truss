#ifndef CUDAKERNEL_H
#define CUDAKERNEL_H

#include "Edge.h"

void kernel(double *A, double *B, double *C, int arraySize);

int* graphKernel(const EdgeList& edges, int n, int* flatarray, int flatarraylength, int* indices);

#endif