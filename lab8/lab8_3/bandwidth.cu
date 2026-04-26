#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// Размер блока для копирования: 100 МБ
const size_t BLOCK_SIZE = 100 * 1024 * 1024;
const int REPEATS = 5; // число повторов для усреднения

// Печать пропускной способности (ГБ/с)
static void print_bw(const char *name, float ms, size_t bytes)
{
    double seconds = ms / 1000.0;
    double gb = (double)bytes / (1024.0 * 1024.0 * 1024.0);
    cout << name << ": time = " << ms << " ms, bandwidth = "
         << (gb / seconds) << " GB/s\n";
}

// Проверка корректности копирования (эталон записан 0xAB)
static bool verify(unsigned char *buf, size_t n, unsigned char pattern)
{
    for (size_t i = 0; i < n; i++)
        if (buf[i] != pattern) return false;
    return true;
}

int main()
{
    cout << "CUDA memory bandwidth test (block size = "
         << (BLOCK_SIZE / (1024 * 1024)) << " MB)\n\n";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cout << device_count << " CUDA device(s) found\n";
    if (device_count == 0) { cout << "No CUDA device found.\n"; return 1; }

    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);
    cout << "GPU 0: " << dp.name << "\n\n";

    // События для замеров
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------- 1. RAM -> RAM ----------
    {
        unsigned char *src = (unsigned char*)malloc(BLOCK_SIZE);
        unsigned char *dst = (unsigned char*)malloc(BLOCK_SIZE);
        memset(src, 0xAB, BLOCK_SIZE);
        memset(dst, 0x00, BLOCK_SIZE);

        float total_ms = 0;
        for (int r = 0; r < REPEATS; r++) {
            cudaEventRecord(start);
            cudaMemcpy(dst, src, BLOCK_SIZE, cudaMemcpyHostToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        cout << "Host -> Host (RAM -> RAM) correct: "
             << (verify(dst, BLOCK_SIZE, 0xAB) ? "YES" : "NO") << "\n";
        print_bw("Host -> Host", total_ms / REPEATS, BLOCK_SIZE);
        cout << "\n";
        free(src); free(dst);
    }

    // ---------- 2. Host (pageable) <-> Device ----------
    {
        unsigned char *h = (unsigned char*)malloc(BLOCK_SIZE);
        unsigned char *d = NULL;
        cudaMalloc((void**)&d, BLOCK_SIZE);
        memset(h, 0xAB, BLOCK_SIZE);

        // H->D
        float total_ms = 0;
        for (int r = 0; r < REPEATS; r++) {
            cudaEventRecord(start);
            cudaMemcpy(d, h, BLOCK_SIZE, cudaMemcpyHostToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0; cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        print_bw("Host -> Device (pageable)", total_ms / REPEATS, BLOCK_SIZE);

        // D->H
        memset(h, 0, BLOCK_SIZE);
        total_ms = 0;
        for (int r = 0; r < REPEATS; r++) {
            cudaEventRecord(start);
            cudaMemcpy(h, d, BLOCK_SIZE, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0; cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        cout << "Device -> Host (pageable) correct: "
             << (verify(h, BLOCK_SIZE, 0xAB) ? "YES" : "NO") << "\n";
        print_bw("Device -> Host (pageable)", total_ms / REPEATS, BLOCK_SIZE);
        cout << "\n";

        free(h); cudaFree(d);
    }

    // ---------- 3. Host (page-locked) <-> Device ----------
    {
        unsigned char *h = NULL;
        cudaMallocHost((void**)&h, BLOCK_SIZE);
        unsigned char *d = NULL;
        cudaMalloc((void**)&d, BLOCK_SIZE);
        memset(h, 0xAB, BLOCK_SIZE);

        // H->D
        float total_ms = 0;
        for (int r = 0; r < REPEATS; r++) {
            cudaEventRecord(start);
            cudaMemcpy(d, h, BLOCK_SIZE, cudaMemcpyHostToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0; cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        print_bw("Host -> Device (page-locked)", total_ms / REPEATS, BLOCK_SIZE);

        // D->H
        memset(h, 0, BLOCK_SIZE);
        total_ms = 0;
        for (int r = 0; r < REPEATS; r++) {
            cudaEventRecord(start);
            cudaMemcpy(h, d, BLOCK_SIZE, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0; cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }
        cout << "Device -> Host (page-locked) correct: "
             << (verify(h, BLOCK_SIZE, 0xAB) ? "YES" : "NO") << "\n";
        print_bw("Device -> Host (page-locked)", total_ms / REPEATS, BLOCK_SIZE);
        cout << "\n";

        cudaFreeHost(h); cudaFree(d);
    }

    // ---------- 4. Device -> Device ----------
    {
        unsigned char *d1 = NULL, *d2 = NULL;
        cudaMalloc((void**)&d1, BLOCK_SIZE);
        cudaMalloc((void**)&d2, BLOCK_SIZE);

        // Заполняем d1 через page-locked для скорости
        unsigned char *h = NULL;
        cudaMallocHost((void**)&h, BLOCK_SIZE);
        memset(h, 0xAB, BLOCK_SIZE);
        cudaMemcpy(d1, h, BLOCK_SIZE, cudaMemcpyHostToDevice);

        float total_ms = 0;
        for (int r = 0; r < REPEATS; r++) {
            cudaEventRecord(start);
            cudaMemcpy(d2, d1, BLOCK_SIZE, cudaMemcpyDeviceToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms = 0; cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }

        // Проверка: копируем d2 обратно в host
        memset(h, 0, BLOCK_SIZE);
        cudaMemcpy(h, d2, BLOCK_SIZE, cudaMemcpyDeviceToHost);
        cout << "Device -> Device correct: "
             << (verify(h, BLOCK_SIZE, 0xAB) ? "YES" : "NO") << "\n";
        print_bw("Device -> Device", total_ms / REPEATS, BLOCK_SIZE);
        cout << "\n";

        cudaFreeHost(h); cudaFree(d1); cudaFree(d2);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << "Done.\n";
    return 0;
}
