#include<time.h>
#include<iostream>
#include<stdlib.h>

#define NUM_VECS (1024 )
#define VEC_LENGTH 512
#define TOTAL_SIZE NUM_VECS*VEC_LENGTH
#define NUM_BLOCKS 2 
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
    float vecs[TOTAL_SIZE];
    float vecs_sorted_by_gpu[TOTAL_SIZE];

    srand(time(NULL));
    for(int i=0; i<TOTAL_SIZE; i++){
        vecs[i] = ((float) rand())/RAND_MAX;
    }

    float* dev_vecs;
    cudaMalloc((void**) & dev_vecs, TOTAL_SIZE*sizeof(float));
    cudaMemcpy(dev_vecs, vecs, TOTAL_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    clock_t t1 = clock();
    sort_gpu<<<NUM_BLOCKS, NUM_THREADS>>>(dev_vecs, VEC_LENGTH);
    clock_t t2 = clock();
    cout << "GPU time = " << ((double)(t2-t1)) / CLOCKS_PER_SEC << '\n';

    clock_t t3 = clock();
    sort_cpu(vecs, VEC_LENGTH, NUM_VECS);
    clock_t t4 = clock();
    cout << "CPU time = " << ((double)(t4-t3)) / CLOCKS_PER_SEC << '\n';

    cudaMemcpy(vecs_sorted_by_gpu, dev_vecs, TOTAL_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    char pass_test = 1;
    for(int i=0; i<NUM_VECS; i++){
        for(int j=0; j<VEC_LENGTH; j++){
            if(vecs_sorted_by_gpu[i*VEC_LENGTH+j] != vecs[i*VEC_LENGTH+j]){
                cout << "GPU and CPU results differs at " << i*VEC_LENGTH+j;
                pass_test = 0;
            }
        }
    }

    if(pass_test){
        cout << "GPU and CPU yeild the same result\n";
    }
}
