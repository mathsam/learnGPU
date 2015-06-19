#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define VEC_LENGTH 20*1024*32
#define NUM_THREADS 32 

__global__ void add(float* dev_a, float* dev_b, float* dev_c){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < VEC_LENGTH){
        dev_c[tid] = exp(sin(dev_a[tid]) + sin(dev_b[tid]));
    }
}

int main(void){
    float a[VEC_LENGTH];
    float b[VEC_LENGTH];
    float c[VEC_LENGTH];

    for(int i=0; i<VEC_LENGTH; i++){
        a[i] = (float) rand() / (float) 0xffffffff;
        b[i] = (float) rand() / (float) 0xffffffff;
    }

    clock_t begin, end;
    begin = clock();
    for(int i=0; i<VEC_LENGTH; i++){
        c[i] = exp(sin(a[i]) + sin(b[i]));
    }
    end = clock();
    printf("time spent by CPU is %f sec\n", (double)(end - begin) / CLOCKS_PER_SEC);

    begin = clock();
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **)& dev_a, VEC_LENGTH*sizeof(float));
    cudaMalloc((void **)& dev_b, VEC_LENGTH*sizeof(float));
    cudaMalloc((void **)& dev_c, VEC_LENGTH*sizeof(float));
    cudaMemcpy(dev_a, a, VEC_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, VEC_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
    add<<<VEC_LENGTH/NUM_THREADS, NUM_THREADS>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(dev_c, c, VEC_LENGTH*sizeof(float), cudaMemcpyDeviceToHost);
    end = clock(); 
    printf("time spent by GPU is %f sec\n", (double)(end - begin) / CLOCKS_PER_SEC);

    return 0;
}
