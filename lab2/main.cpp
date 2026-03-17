#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>

const size_t N = 2048;
const float EPSILON = 1e-3f;

using Matrix = std::vector<std::vector<float>>;

void multiplyClassic(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float s = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
}

Matrix transpose(const Matrix& B) {
    size_t n = B.size();
    Matrix Bt(n, std::vector<float>(n));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            Bt[j][i] = B[i][j];
        }
    }
    return Bt;
}

void multiplyTranspose(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t n = A.size();
    Matrix Bt = transpose(B);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float s = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                s += A[i][k] * Bt[j][k];
            }
            C[i][j] = s;
        }
    }
}

void multiplyBuffered(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t n = A.size();
    std::vector<float> tmp(n);

    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
            tmp[k] = B[k][j];
        }

        for (size_t i = 0; i < n; ++i) {
            float s = 0.0f;
            for (size_t k = 0; k < n; ++k) {
                s += A[i][k] * tmp[k];
            }
            C[i][j] = s;
        }
    }
}

void multiplyBlock(const Matrix& A, const Matrix& B, Matrix& C, int blockSize) {
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        std::fill(C[i].begin(), C[i].end(), 0.0f);
    }

    for (size_t ii = 0; ii < n; ii += blockSize) {
        for (size_t jj = 0; jj < n; jj += blockSize) {
            for (size_t kk = 0; kk < n; kk += blockSize) {
                size_t iMax = std::min(ii + blockSize, n);
                size_t jMax = std::min(jj + blockSize, n);
                size_t kMax = std::min(kk + blockSize, n);

                for (size_t i = ii; i < iMax; ++i) {
                    for (size_t j = jj; j < jMax; ++j) {
                        float s = 0.0f;
                        for (size_t k = kk; k < kMax; ++k) {
                            s += A[i][k] * B[k][j];
                        }
                        C[i][j] += s;
                    }
                }
            }
        }
    }
}

template<int UNROLL>
void multiplyBufferedUnrolled(const Matrix& A, const Matrix& B, Matrix& C) {
    size_t n = A.size();
    std::vector<float> tmp(n);

    for (size_t j = 0; j < n; ++j) {
        for (size_t k = 0; k < n; ++k) {
            tmp[k] = B[k][j];
        }

        for (size_t i = 0; i < n; ++i) {
            float s[UNROLL] = {0.0f};
            size_t k;

            for (k = 0; k + UNROLL - 1 < n; k += UNROLL) {
                for (int u = 0; u < UNROLL; ++u) {
                    s[u] += A[i][k + u] * tmp[k + u];
                }
            }

            float total = 0.0f;
            for (int u = 0; u < UNROLL; ++u) {
                total += s[u];
            }
            for (; k < n; ++k) {
                total += A[i][k] * tmp[k];
            }

            C[i][j] = total;
        }
    }
}

Matrix createMatrix(size_t n) {
    return Matrix(n, std::vector<float>(n));
}

void fillRandom(Matrix& mat) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

    for (auto& row : mat) {
        for (auto& val : row) {
            val = dis(gen);
        }
    }
}

template <class> struct ExeTime;

template <class R, class... Args>
struct ExeTime<R(Args...)> {
public:
    ExeTime(std::function<R(Args...)> func) : f_(func) {}

    std::pair<R, double> operator()(Args... args) {
        auto start = std::chrono::high_resolution_clock::now();
        R result = f_(args...);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double seconds = duration.count() / 1'000'000.0;

        return {result, seconds};
    }

private:
    std::function<R(Args...)> f_;
};

template <class... Args>
struct ExeTime<void(Args...)> {
public:
    ExeTime(std::function<void(Args...)> func) : f_(func) {}

    double operator()(Args... args) {
        auto start = std::chrono::high_resolution_clock::now();
        f_(args...);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return duration.count() / 1'000'000.0;
    }

private:
    std::function<void(Args...)> f_;
};

template <class R, class... Args>
ExeTime<R(Args...)> make_decorator(R (*f)(Args...)) {
    return ExeTime<R(Args...)>(std::function<R(Args...)>(f));
}

template <class... Args>
ExeTime<void(Args...)> make_decorator(void (*f)(Args...)) {
    return ExeTime<void(Args...)>(std::function<void(Args...)>(f));
}

int main() {
    return 0;
}
