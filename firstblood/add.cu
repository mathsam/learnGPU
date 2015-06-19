#include <stdio.h>
#include <stdlib.h>

__global__ void what_is_my_id(unsigned int * const block,
    unsigned int * const thread,
    unsigned int * const warp,
    unsigned int * const calc_thread)
{
  /* Thread id is block index * block size + thread offset into the block */
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  block[thread_idx] = blockIdx.x;
  thread[thread_idx] = threadIdx.x;

  /* Calculate warp using built in variable warpSize */
  warp[thread_idx] = threadIdx.x / warpSize;
  calc_thread[thread_idx] = thread_idx;
}

__global__ void what_is_my_id_2d(
    unsigned int * const block_x,
    unsigned int * const block_y,
    unsigned int * const calc_thread)
{
  unsigned int idx = gridDim.x * blockDim.x * (blockIdx.y * blockDim.y + threadIdx.y)
                   + blockIdx.x * blockDim.x + threadIdx.x;
  block_x[idx] = blockIdx.x;
  block_y[idx] = blockIdx.y;
  calc_thread[idx] = idx;
}
    

#define ARRAY_SIZE 128*10
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))
#define SIZE_X 32
#define SIZE_Y 16
#define SIZE2D_BYTES (sizeof(unsigned int) * SIZE_X * SIZE_Y)
/* Declare statically four arrays of ARRAY_SIZE each */
unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

unsigned int cpu_block_x_2d[SIZE_Y][SIZE_X];
unsigned int cpu_block_y_2d[SIZE_Y][SIZE_X];
unsigned int cpu_calc_thread_2d[SIZE_Y][SIZE_X];

int main(void)
{
  /* Total thread count = 2 * 64 = 128 */
  const unsigned int num_blocks = 2*10;
  const unsigned int num_threads = 64;

  /* Declare pointers for GPU based params */
  unsigned int * gpu_block;
  unsigned int * gpu_thread;
  unsigned int * gpu_warp;
  unsigned int * gpu_calc_thread;

  unsigned int * gpu_block_x_2d;
  unsigned int * gpu_block_y_2d;
  unsigned int * gpu_calc_thread_2d;

  /* Declare loop counter for use later */
  unsigned int i;
  
  /* Allocate four arrays on the GPU */
/*
  cudaMalloc((void **)&gpu_block, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_thread, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_warp, ARRAY_SIZE_IN_BYTES);
  cudaMalloc((void **)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
*/

  cudaMalloc((void **)&gpu_block_x_2d, SIZE2D_BYTES);
  cudaMalloc((void **)&gpu_block_y_2d, SIZE2D_BYTES);
  cudaMalloc((void **)&gpu_calc_thread_2d, SIZE2D_BYTES);

  const dim3 threads_square(16, 8);
  const dim3 blocks_square(2, 2);

  /* Execute our kernel */
//  what_is_my_id<<<num_blocks, num_threads>>>(gpu_block, gpu_thread, gpu_warp,
//    gpu_calc_thread);

  what_is_my_id_2d<<<blocks_square, threads_square>>>(gpu_block_x_2d, 
    gpu_block_y_2d, gpu_calc_thread_2d);

  /* Copy back the gpu results to the CPU */
/*
  cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES,
    cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES,
    cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES,
    cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES,
    cudaMemcpyDeviceToHost);
*/
  cudaMemcpy(cpu_block_x_2d, gpu_block_x_2d, SIZE2D_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_block_y_2d, gpu_block_y_2d, SIZE2D_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(cpu_calc_thread_2d, gpu_calc_thread_2d, SIZE2D_BYTES, cudaMemcpyDeviceToHost);

  /* Free the arrays on the GPU as now we’re done with them */
  
/*
  cudaFree(gpu_block);
  cudaFree(gpu_thread);
  cudaFree(gpu_warp);
  cudaFree(gpu_calc_thread);
*/
  cudaFree(gpu_block_x_2d);
  cudaFree(gpu_block_y_2d);
  cudaFree(gpu_calc_thread_2d);

  /* Iterate through the arrays and print */
/*
  for (unsigned int i=0; i < ARRAY_SIZE; i++)
  {
    printf("Calculated Thread: %3u - Block: %2u - Warp %2u - Thread %3u\n",
      cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
  }
*/
  for (i=0; i < SIZE_X*SIZE_Y; i++)
  {
    printf("Calculated Thread: %3u - Block X: %2u - Block Y %2u\n",
      *(&cpu_calc_thread_2d[0][0] + i), *(&cpu_block_x_2d[0][0] +i), *(&cpu_block_y_2d[0][0] + i));
  }

}

