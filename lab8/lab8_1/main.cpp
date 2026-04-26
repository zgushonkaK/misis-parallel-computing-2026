#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

extern "C" void vec_mul_cuda(float *a, float *b, float *c, int n);
extern "C" void vec_add_cuda(float *a, float *b, float *c, int n);

const int N = 1024;

float a[N], b[N];
float c_gpu_mul[N], c_cpu_mul[N];
float c_gpu_add[N], c_cpu_add[N];

void vec_mul_cpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] * b[i];
}

void vec_add_cpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

bool compare(float *x, float *y, int n, float eps = 1e-4f)
{
    for (int i = 0; i < n; i++)
        if (fabs(x[i] - y[i]) > eps)
            return false;
    return true;
}

int main()
{
    srand((unsigned)time(NULL));

    for (int i = 0; i < N; i++)
    {
        a[i] = (float)(rand() % 100) / 10.0f;
        b[i] = (float)(rand() % 100) / 10.0f;
    }

    vec_mul_cuda(a, b, c_gpu_mul, N);
    vec_mul_cpu (a, b, c_cpu_mul, N);

    cout << "Multiplication (first 10 elements):\n";
    cout << "GPU: ";
    for (int i = 0; i < 10; i++) cout << c_gpu_mul[i] << " ";
    cout << "\nCPU: ";
    for (int i = 0; i < 10; i++) cout << c_cpu_mul[i] << " ";
    cout << "\nMul match: " << (compare(c_gpu_mul, c_cpu_mul, N) ? "YES" : "NO") << "\n\n";

    vec_add_cuda(a, b, c_gpu_add, N);
    vec_add_cpu (a, b, c_cpu_add, N);

    cout << "Addition (first 10 elements):\n";
    cout << "GPU: ";
    for (int i = 0; i < 10; i++) cout << c_gpu_add[i] << " ";
    cout << "\nCPU: ";
    for (int i = 0; i < 10; i++) cout << c_cpu_add[i] << " ";
    cout << "\nAdd match: " << (compare(c_gpu_add, c_cpu_add, N) ? "YES" : "NO") << "\n";

    cout << "\nDone.\n";
    return 0;
}
