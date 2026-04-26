#include <cuda.h>
#include <cuda_runtime.h>

// Ядро поэлементного умножения векторов
__global__ void VecMulKernel(float *a, float *b, float *c)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] * b[i];
}

// Ядро поэлементного сложения векторов (второй пункт задания)
__global__ void VecAddKernel(float *a, float *b, float *c)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

// Обертка: умножение векторов на GPU
extern "C" void vec_mul_cuda(float *a, float *b, float *c, int n)
{
    int SizeInBytes = n * sizeof(float);

    float *a_gpu = NULL;
    float *b_gpu = NULL;
    float *c_gpu = NULL;

    cudaMalloc((void **)&a_gpu, SizeInBytes);
    cudaMalloc((void **)&b_gpu, SizeInBytes);
    cudaMalloc((void **)&c_gpu, SizeInBytes);

    cudaMemcpy(a_gpu, a, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, SizeInBytes, cudaMemcpyHostToDevice);

    dim3 threads = dim3(512, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    VecMulKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);

    cudaMemcpy(c, c_gpu, SizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}

// Обертка: сложение векторов на GPU
extern "C" void vec_add_cuda(float *a, float *b, float *c, int n)
{
    int SizeInBytes = n * sizeof(float);

    float *a_gpu = NULL;
    float *b_gpu = NULL;
    float *c_gpu = NULL;

    cudaMalloc((void **)&a_gpu, SizeInBytes);
    cudaMalloc((void **)&b_gpu, SizeInBytes);
    cudaMalloc((void **)&c_gpu, SizeInBytes);

    cudaMemcpy(a_gpu, a, SizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu, b, SizeInBytes, cudaMemcpyHostToDevice);

    dim3 threads = dim3(512, 1);
    dim3 blocks  = dim3(n / threads.x, 1);

    VecAddKernel<<<blocks, threads>>>(a_gpu, b_gpu, c_gpu);

    cudaMemcpy(c, c_gpu, SizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
}
