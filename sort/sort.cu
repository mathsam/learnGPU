#include<time.h>
#include<stdlib.h>
#include<stdio.h>

#define NUM_VECS (1024*128)
#define VEC_LENGTH 512
#define TOTAL_SIZE NUM_VECS*VEC_LENGTH
#define NUM_BLOCKS 256 
#define NUM_THREADS NUM_VECS/NUM_BLOCKS

using namespace std;

__global__ void sort_gpu(float* vecs, unsigned int length){
    unsigned int start = (blockIdx.x*blockDim.x + threadIdx.x)*length;
    for(unsigned int i = start+1; i < start+length; i++){
        unsigned int j = i;
        while(j > start && vecs[j] < vecs[j-1]){
            float tmp = vecs[j];
            vecs[j] = vecs[j-1];
            vecs[j-1] = tmp;
            j--;
        }
    }
}

void sort_cpu(float* vecs, unsigned int length, unsigned int num_vecs){
    for(unsigned i_vec = 0; i_vec < num_vecs; i_vec++){
        unsigned int start = i_vec*length;
        for(unsigned int i = start+1; i < start+length; i++){
            unsigned int j = i;
            while(j > start && vecs[j] < vecs[j-1]){
                float tmp = vecs[j];
                vecs[j] = vecs[j-1];
                vecs[j-1] = tmp;
                j--;
            }
        }
    }
}

int main(void){
    float* vecs = new float[TOTAL_SIZE];
    float* vecs_sorted_by_gpu = new float[TOTAL_SIZE];

    srand(time(NULL));
    for(int i=0; i<TOTAL_SIZE; i++){
        vecs[i] = ((float) rand())/RAND_MAX;
    }

    float* dev_vecs;
    cudaError_t error;
    error = cudaMalloc((void**) & dev_vecs, TOTAL_SIZE*sizeof(float));
    printf("allocation mem on device: %s\n", cudaGetErrorString(error));
    cudaMemcpy(dev_vecs, vecs, TOTAL_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    clock_t t1 = clock();
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    sort_gpu<<<NUM_BLOCKS, NUM_THREADS>>>(dev_vecs, VEC_LENGTH);

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop);
    float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime, start, stop); 
    printf("time required : %f ms", elapsedTime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(vecs_sorted_by_gpu, dev_vecs, TOTAL_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    size_t free, total;
    printf("\n");
    cudaMemGetInfo(&free,&total); 
    printf("%4f MB free of total %4f MB\n",(float)free/1024./1024.,total/1024./1024.);

    /*
    clock_t t3 = clock();
    sort_cpu(vecs, VEC_LENGTH, NUM_VECS);
    clock_t t4 = clock();
    printf("CPU time = %f (ms)", 1000.*((double)(t4-t3)) / CLOCKS_PER_SEC);


    char pass_test = 1;
    for(int i=0; i<NUM_VECS; i++){
        for(int j=0; j<VEC_LENGTH; j++){
            if(vecs_sorted_by_gpu[i*VEC_LENGTH+j] != vecs[i*VEC_LENGTH+j]){
                printf("GPU and CPU results differs at %d", i*VEC_LENGTH+j);
                pass_test = 0;
            }
        }
    }

    if(pass_test){
        printf("GPU and CPU yeild the same result\n");
    }
    */

    for(int i=0; i<NUM_VECS; i++){
        for(int j=0; j<VEC_LENGTH; j++){
            printf("%4f, ", vecs_sorted_by_gpu[i*VEC_LENGTH+j]);
            }
        printf("\n\n");
    }
}
