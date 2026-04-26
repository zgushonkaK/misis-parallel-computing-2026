#include <cuda.h>
#include <cuda_runtime.h>

// =====================================================================
// 1. Наивное (классическое) умножение — каждый поток один элемент C
// =====================================================================
__global__ void MatrMulKernel(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // строка
    int j = blockIdx.x * blockDim.x + threadIdx.x; // столбец
    if (i >= N || j >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += A[i * N + k] * B[k * N + j];
    C[i * N + j] = sum;
}

// =====================================================================
// 2. Кэширование строки матрицы A в разделяемой памяти
//    Запуск: <<<dim3(N), dim3(N)>>> — один блок на строку C
// =====================================================================
__global__ void MatrMulRowCacheKernel(const float *A, const float *B, float *C, int N)
{
    extern __shared__ float rowA[]; // размер N float'ов

    int i = blockIdx.x;     // строка
    int j = threadIdx.x;    // столбец

    // Все потоки блока совместно грузят строку A[i, *] в shared
    for (int k = j; k < N; k += blockDim.x)
        rowA[k] = A[i * N + k];
    __syncthreads();

    if (j < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += rowA[k] * B[k * N + j];
        C[i * N + j] = sum;
    }
}

// =====================================================================
// 3. Кэширование столбца матрицы B в разделяемой памяти
//    Запуск: <<<dim3(N), dim3(N)>>> — один блок на столбец C
// =====================================================================
__global__ void MatrMulColCacheKernel(const float *A, const float *B, float *C, int N)
{
    extern __shared__ float colB[]; // размер N float'ов

    int j = blockIdx.x;     // столбец
    int i = threadIdx.x;    // строка

    for (int k = i; k < N; k += blockDim.x)
        colB[k] = B[k * N + j];
    __syncthreads();

    if (i < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[i * N + k] * colB[k];
        C[i * N + j] = sum;
    }
}

// =====================================================================
// 4. Блочное умножение через разделяемую память
//    Запуск: <<<dim3(N/S, N/S), dim3(S, S)>>>
//    Размер блока S задаётся через шаблон
// =====================================================================
template <int S>
__global__ void MatrMulBlockKernel(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[S][S];
    __shared__ float Bs[S][S];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * S + ty; // глобальная строка элемента C
    int col = bx * S + tx; // глобальный столбец элемента C

    float sum = 0.0f;

    // Проход по подматрицам
    for (int z = 0; z < N / S; z++) {
        // Загрузка блока A[by, z] и B[z, bx] в shared
        As[ty][tx] = A[row * N + (z * S + tx)];
        Bs[ty][tx] = B[(z * S + ty) * N + col];
        __syncthreads();

        // Умножение блоков
        #pragma unroll
        for (int k = 0; k < S; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }
    C[row * N + col] = sum;
}

// =====================================================================
// 5. Блочное умножение с раскруткой цикла (UNROLL = 2/4/8 внутри #pragma unroll)
//    Здесь отдельная версия, где unroll явный — используется то же тело,
//    но с указанием шага раскрутки параметром шаблона UR.
// =====================================================================
template <int S, int UR>
__global__ void MatrMulBlockKernelUnroll(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[S][S];
    __shared__ float Bs[S][S];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * S + ty;
    int col = bx * S + tx;

    float sum = 0.0f;
    for (int z = 0; z < N / S; z++) {
        As[ty][tx] = A[row * N + (z * S + tx)];
        Bs[ty][tx] = B[(z * S + ty) * N + col];
        __syncthreads();

        #pragma unroll UR
        for (int k = 0; k < S; k++)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }
    C[row * N + col] = sum;
}

// Явные инстанциации шаблонов для используемых размеров блока
template __global__ void MatrMulBlockKernel<8> (const float*, const float*, float*, int);
template __global__ void MatrMulBlockKernel<16>(const float*, const float*, float*, int);
template __global__ void MatrMulBlockKernel<32>(const float*, const float*, float*, int);

template __global__ void MatrMulBlockKernelUnroll<16, 2>(const float*, const float*, float*, int);
template __global__ void MatrMulBlockKernelUnroll<16, 4>(const float*, const float*, float*, int);
template __global__ void MatrMulBlockKernelUnroll<16, 8>(const float*, const float*, float*, int);
template __global__ void MatrMulBlockKernelUnroll<32, 2>(const float*, const float*, float*, int);
template __global__ void MatrMulBlockKernelUnroll<32, 4>(const float*, const float*, float*, int);
template __global__ void MatrMulBlockKernelUnroll<32, 8>(const float*, const float*, float*, int);
