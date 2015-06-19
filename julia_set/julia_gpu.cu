#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <time.h>
#include <iostream>

#define DIM 1024
#define NUM_THREADS 128 

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex( float a, float b ) : r(a), i(b)  {}

    __device__ float magnitude2( void ) { return r * r + i * i; }

    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};


__device__ int julia( int x, int y, float param_r, float param_i) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(param_r, param_i);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel(unsigned char *ptr, float param_r, float param_i){
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < DIM*DIM){
        unsigned x = idx%DIM;
        unsigned y = idx/DIM;
        ptr[4*idx + 0] = 150 * julia(x, y, param_r, param_i);
        ptr[4*idx + 1] = 30;
        ptr[4*idx + 2] = 90;
        ptr[4*idx + 3] = 67;
    }
}

int main(void){
    CPUBitmap bitmap( DIM, DIM );
    unsigned char *ptr = bitmap.get_ptr();

    float param_r, param_i;
    std::cout << "Input params:\n";
    std::cout << "Real part: ";
    std::cin >> param_r;
    std::cout << "\nImag part: ";
    std::cin >> param_i;

    clock_t begin, end;
    begin = clock();

    unsigned char *dev_ptr;
    cudaMalloc((void**) & dev_ptr, bitmap.image_size());

/*
    float * dev_param_r;
    float * dev_param_i;
    cudaMalloc((void**) & dev_param_r, sizeof(float));
    cudaMalloc((void**) & dev_param_i, sizeof(float));
    cudaMemcpy(dev_param_r, &param_r, sizeof(float), cudaMemcpyHostToDevice);    cudaMemcpy(dev_param_i, &param_i, sizeof(float), cudaMemcpyHostToDevice);
*/

    kernel<<<DIM*DIM/NUM_THREADS, NUM_THREADS>>>(dev_ptr, param_r, param_i);
    cudaMemcpy(ptr, dev_ptr, bitmap.image_size(), cudaMemcpyDeviceToHost);

    cudaFree(dev_ptr);
    end = clock();
    printf("time spent by GPU is %f sec\n", (double)(end - begin) / CLOCKS_PER_SEC); 

    bitmap.display_and_exit();

    return 0;
}
