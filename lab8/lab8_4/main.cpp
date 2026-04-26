#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <fstream>
#include <vector>

using namespace std;

// Внешние функции из .cu
extern "C" void  matmul_cpu(const float*, const float*, float*, int);
extern "C" float matmul_gpu_naive(const float*, const float*, float*, int, int);
extern "C" float matmul_gpu_row  (const float*, const float*, float*, int, int);
extern "C" float matmul_gpu_col  (const float*, const float*, float*, int, int);
extern "C" float matmul_gpu_block(const float*, const float*, float*, int, int);
extern "C" float matmul_gpu_block_unroll(const float*, const float*, float*, int, int, int);
extern "C" const char* cuda_last_error_str();

// Заполнение случайными данными
static void fill_rand(float *M, int N)
{
    for (int i = 0; i < N * N; i++)
        M[i] = (float)(rand() % 100) / 10.0f;
}

// Проверка равенства двух матриц с допуском
static bool matrices_equal(const float *A, const float *B, int N, float eps = 1e-2f)
{
    for (int i = 0; i < N * N; i++) {
        float diff = fabs(A[i] - B[i]);
        float ref  = fabs(A[i]) + 1.0f;
        if (diff / ref > eps) {
            cerr << "Mismatch at " << i << ": " << A[i] << " vs " << B[i] << "\n";
            return false;
        }
    }
    return true;
}

// Реальная производительность в GFLOPS
static double gflops(int N, float ms)
{
    double ops = 2.0 * (double)N * (double)N * (double)N; // 2*N^3
    return ops / (ms / 1000.0) / 1e9;
}

int main(int argc, char **argv)
{
    srand((unsigned)time(NULL));

    // Размеры матриц для тестирования (степени двойки)
    vector<int> sizes = {256, 512, 1024};

    ofstream csv("results.csv");
    csv << "algorithm,N,S,UR,time_ms,gflops,correct\n";

    cout << fixed << setprecision(3);

    for (int N : sizes) {
        cout << "\n========================= N = " << N << " =========================\n";
        size_t bytes = (size_t)N * N * sizeof(float);

        float *A = (float*)malloc(bytes);
        float *B = (float*)malloc(bytes);
        float *C_cpu    = (float*)malloc(bytes);
        float *C_gpu    = (float*)malloc(bytes);

        fill_rand(A, N);
        fill_rand(B, N);

        // ---- CPU эталон (только для небольших N, чтобы не ждать полчаса) ----
        bool run_cpu = (N <= 1024);
        double cpu_ms = 0.0;
        if (run_cpu) {
            auto t0 = chrono::high_resolution_clock::now();
            matmul_cpu(A, B, C_cpu, N);
            auto t1 = chrono::high_resolution_clock::now();
            cpu_ms = chrono::duration<double, milli>(t1 - t0).count();
            cout << "CPU:           time = " << cpu_ms << " ms, "
                 << gflops(N, (float)cpu_ms) << " GFLOPS\n";
            csv << "cpu," << N << ",,," << cpu_ms << "," << gflops(N, (float)cpu_ms) << ",1\n";
        }

        // ---- Наивный GPU ----
        for (int bd : {8, 16, 32}) {
            memset(C_gpu, 0, bytes);
            float ms = matmul_gpu_naive(A, B, C_gpu, N, bd);
            bool ok = run_cpu ? matrices_equal(C_cpu, C_gpu, N) : true;
            double gf = gflops(N, ms);
            cout << "GPU naive      (bd=" << bd << "): time = " << ms
                 << " ms, " << gf << " GFLOPS, "
                 << (ok ? "OK" : "FAIL")
                 << " [err: " << cuda_last_error_str() << "]\n";
            csv << "naive," << N << "," << bd << ",," << ms << "," << gf
                << "," << (ok ? 1 : 0) << "\n";
        }

        // ---- Кэш строки ----
        // Варьируем число потоков в блоке
        for (int tpb : {64, 128, 256, 512}) {
            if (tpb > N) continue;
            memset(C_gpu, 0, bytes);
            float ms = matmul_gpu_row(A, B, C_gpu, N, tpb);
            bool ok = run_cpu ? matrices_equal(C_cpu, C_gpu, N) : true;
            double gf = gflops(N, ms);
            cout << "GPU row-cache  (tpb=" << tpb << "): time = " << ms
                 << " ms, " << gf << " GFLOPS, "
                 << (ok ? "OK" : "FAIL") << "\n";
            csv << "row," << N << "," << tpb << ",," << ms << "," << gf
                << "," << (ok ? 1 : 0) << "\n";
        }

        // ---- Кэш столбца ----
        for (int tpb : {64, 128, 256, 512}) {
            if (tpb > N) continue;
            memset(C_gpu, 0, bytes);
            float ms = matmul_gpu_col(A, B, C_gpu, N, tpb);
            bool ok = run_cpu ? matrices_equal(C_cpu, C_gpu, N) : true;
            double gf = gflops(N, ms);
            cout << "GPU col-cache  (tpb=" << tpb << "): time = " << ms
                 << " ms, " << gf << " GFLOPS, "
                 << (ok ? "OK" : "FAIL") << "\n";
            csv << "col," << N << "," << tpb << ",," << ms << "," << gf
                << "," << (ok ? 1 : 0) << "\n";
        }

        // ---- Блочное ----
        for (int S : {8, 16, 32}) {
            if (N % S != 0) continue;
            memset(C_gpu, 0, bytes);
            float ms = matmul_gpu_block(A, B, C_gpu, N, S);
            bool ok = run_cpu ? matrices_equal(C_cpu, C_gpu, N) : true;
            double gf = gflops(N, ms);
            cout << "GPU block      (S=" << S << "):   time = " << ms
                 << " ms, " << gf << " GFLOPS, "
                 << (ok ? "OK" : "FAIL") << "\n";
            csv << "block," << N << "," << S << ",," << ms << "," << gf
                << "," << (ok ? 1 : 0) << "\n";
        }

        // ---- Блочное с unroll ----
        for (int S : {16, 32}) {
            if (N % S != 0) continue;
            for (int ur : {2, 4, 8}) {
                memset(C_gpu, 0, bytes);
                float ms = matmul_gpu_block_unroll(A, B, C_gpu, N, S, ur);
                bool ok = run_cpu ? matrices_equal(C_cpu, C_gpu, N) : true;
                double gf = gflops(N, ms);
                cout << "GPU block-unr  (S=" << S << ", UR=" << ur << "): time = "
                     << ms << " ms, " << gf << " GFLOPS, "
                     << (ok ? "OK" : "FAIL") << "\n";
                csv << "block_unroll," << N << "," << S << "," << ur << ","
                    << ms << "," << gf << "," << (ok ? 1 : 0) << "\n";
            }
        }

        free(A); free(B); free(C_cpu); free(C_gpu);
    }

    csv.close();
    cout << "\nResults saved to results.csv\n";
    return 0;
}
