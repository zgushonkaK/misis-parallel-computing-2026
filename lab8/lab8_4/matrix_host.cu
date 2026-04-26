#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

// Объявления ядер (определены в matrix_kernels.cu)
__global__ void MatrMulKernel(const float *A, const float *B, float *C, int N);
__global__ void MatrMulRowCacheKernel(const float *A, const float *B, float *C, int N);
__global__ void MatrMulColCacheKernel(const float *A, const float *B, float *C, int N);

template <int S>
__global__ void MatrMulBlockKernel(const float *A, const float *B, float *C, int N);

template <int S, int UR>
__global__ void MatrMulBlockKernelUnroll(const float *A, const float *B, float *C, int N);

// ---------------- CPU-аналог ----------------
extern "C" void matmul_cpu(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// Вспомогательная функция: копирование на GPU, запуск лямбды, копирование обратно, замер
// Так как CUDA не позволяет красиво лямбды — делаем явные обертки.

static inline float launch_and_time(void (*launch)(const float*, const float*, float*, int),
                                    const float *dA, const float *dB, float *dC, int N)
{
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    launch(dA, dB, dC, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms;
}

// ===== 1. Наивный =====
extern "C" float matmul_gpu_naive(const float *A, const float *B, float *C, int N, int blockDim)
{
    size_t sz = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz); cudaMalloc(&dB, sz); cudaMalloc(&dC, sz);
    cudaMemcpy(dA, A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sz, cudaMemcpyHostToDevice);

    dim3 threads(blockDim, blockDim);
    dim3 blocks((N + blockDim - 1) / blockDim, (N + blockDim - 1) / blockDim);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    MatrMulKernel<<<blocks, threads>>>(dA, dB, dC, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    cudaMemcpy(C, dC, sz, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ms;
}

// ===== 2. Кэш строки =====
extern "C" float matmul_gpu_row(const float *A, const float *B, float *C, int N, int threadsPerBlock)
{
    size_t sz = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz); cudaMalloc(&dB, sz); cudaMalloc(&dC, sz);
    cudaMemcpy(dA, A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sz, cudaMemcpyHostToDevice);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    MatrMulRowCacheKernel<<<N, threadsPerBlock, N * sizeof(float)>>>(dA, dB, dC, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    cudaMemcpy(C, dC, sz, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ms;
}

// ===== 3. Кэш столбца =====
extern "C" float matmul_gpu_col(const float *A, const float *B, float *C, int N, int threadsPerBlock)
{
    size_t sz = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz); cudaMalloc(&dB, sz); cudaMalloc(&dC, sz);
    cudaMemcpy(dA, A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sz, cudaMemcpyHostToDevice);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    MatrMulColCacheKernel<<<N, threadsPerBlock, N * sizeof(float)>>>(dA, dB, dC, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    cudaMemcpy(C, dC, sz, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ms;
}

// ===== 4. Блочное — обертка с параметром размера блока =====
extern "C" float matmul_gpu_block(const float *A, const float *B, float *C, int N, int S)
{
    size_t sz = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz); cudaMalloc(&dB, sz); cudaMalloc(&dC, sz);
    cudaMemcpy(dA, A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sz, cudaMemcpyHostToDevice);

    dim3 threads(S, S);
    dim3 blocks(N / S, N / S);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    if      (S == 8)  MatrMulBlockKernel<8> <<<blocks, threads>>>(dA, dB, dC, N);
    else if (S == 16) MatrMulBlockKernel<16><<<blocks, threads>>>(dA, dB, dC, N);
    else if (S == 32) MatrMulBlockKernel<32><<<blocks, threads>>>(dA, dB, dC, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    cudaMemcpy(C, dC, sz, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ms;
}

// ===== 5. Блочное с раскруткой =====
extern "C" float matmul_gpu_block_unroll(const float *A, const float *B, float *C,
                                         int N, int S, int UR)
{
    size_t sz = (size_t)N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz); cudaMalloc(&dB, sz); cudaMalloc(&dC, sz);
    cudaMemcpy(dA, A, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sz, cudaMemcpyHostToDevice);

    dim3 threads(S, S);
    dim3 blocks(N / S, N / S);

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);

    if (S == 16) {
        if      (UR == 2) MatrMulBlockKernelUnroll<16,2><<<blocks, threads>>>(dA, dB, dC, N);
        else if (UR == 4) MatrMulBlockKernelUnroll<16,4><<<blocks, threads>>>(dA, dB, dC, N);
        else if (UR == 8) MatrMulBlockKernelUnroll<16,8><<<blocks, threads>>>(dA, dB, dC, N);
    } else if (S == 32) {
        if      (UR == 2) MatrMulBlockKernelUnroll<32,2><<<blocks, threads>>>(dA, dB, dC, N);
        else if (UR == 4) MatrMulBlockKernelUnroll<32,4><<<blocks, threads>>>(dA, dB, dC, N);
        else if (UR == 8) MatrMulBlockKernelUnroll<32,8><<<blocks, threads>>>(dA, dB, dC, N);
    }

    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0; cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    cudaMemcpy(C, dC, sz, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return ms;
}

// Проверка последней ошибки CUDA
extern "C" const char* cuda_last_error_str()
{
    cudaError_t err = cudaGetLastError();
    return cudaGetErrorString(err);
}
