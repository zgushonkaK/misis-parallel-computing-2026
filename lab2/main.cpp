#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

const size_t N = 2048;
const float EPSILON = 1e-1f;

using Matrix = std::vector<std::vector<float>>;

struct Results {
  std::vector<std::pair<std::string, double>> basicAlgorithms;
  std::vector<std::pair<int, double>> blockSizes;
  std::vector<std::pair<int, double>> unrollingFactors;
  double transposeTime;
  double multiplyTimeTranspose;
  double classicTime;
  std::vector<std::tuple<int, int, double>> combinedResults;
};

Results globalResults;

void multiplyClassic(const Matrix &A, const Matrix &B, Matrix &C) {
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

Matrix transpose(const Matrix &B) {
  size_t n = B.size();
  Matrix Bt(n, std::vector<float>(n));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      Bt[j][i] = B[i][j];
    }
  }
  return Bt;
}

void multiplyTranspose(const Matrix &A, const Matrix &B, Matrix &C) {
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

void multiplyBuffered(const Matrix &A, const Matrix &B, Matrix &C) {
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

void multiplyBlock(const Matrix &A, const Matrix &B, Matrix &C, int blockSize) {
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

template <int UNROLL>
void multiplyBufferedUnrolled(const Matrix &A, const Matrix &B, Matrix &C) {
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

Matrix createMatrix(size_t n) { return Matrix(n, std::vector<float>(n)); }

void fillRandom(Matrix &mat) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-10.0f, 10.0f);

  for (auto &row : mat) {
    for (auto &val : row) {
      val = dis(gen);
    }
  }
}

void setZero(Matrix &mat) {
  for (auto &row : mat) {
    std::fill(row.begin(), row.end(), 0.0f);
  }
}

bool compareMatrices(const Matrix &A, const Matrix &B) {
  size_t n = A.size();
  if (n != B.size())
    return false;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (std::fabs(A[i][j] - B[i][j]) > EPSILON) {
        return false;
      }
    }
  }
  return true;
}

template <class> struct ExeTime;

template <class R, class... Args> struct ExeTime<R(Args...)> {
public:
  ExeTime(std::function<R(Args...)> func) : f_(func) {}

  std::pair<R, double> operator()(Args... args) {
    auto start = std::chrono::high_resolution_clock::now();
    R result = f_(args...);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double seconds = duration.count() / 1'000'000.0;

    return {result, seconds};
  }

private:
  std::function<R(Args...)> f_;
};

template <class... Args> struct ExeTime<void(Args...)> {
public:
  ExeTime(std::function<void(Args...)> func) : f_(func) {}

  double operator()(Args... args) {
    auto start = std::chrono::high_resolution_clock::now();
    f_(args...);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
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

void measureAndPrint(const std::string &name, const Matrix &A, const Matrix &B,
                     Matrix &C,
                     void (*func)(const Matrix &, const Matrix &, Matrix &)) {
  setZero(C);
  auto decorator = make_decorator(func);
  double seconds = decorator(A, B, C);
  double gflops = (2.0 * N * N * N) / (seconds * 1e9);

  globalResults.basicAlgorithms.push_back({name, gflops});

  std::cout << std::left << std::setw(40) << name << " time: " << std::setw(8)
            << std::fixed << std::setprecision(2) << seconds << " s"
            << " (" << std::setw(6) << std::fixed << std::setprecision(2)
            << gflops << " GFLOPS)" << std::endl;
}

void measureAndPrintWithParam(
    const std::string &name, const Matrix &A, const Matrix &B, Matrix &C,
    void (*func)(const Matrix &, const Matrix &, Matrix &, int), int param) {
  setZero(C);
  auto start = std::chrono::high_resolution_clock::now();
  func(A, B, C, param);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double seconds = duration.count() / 1'000'000.0;
  double gflops = (2.0 * N * N * N) / (seconds * 1e9);

  std::cout << std::left << std::setw(35)
            << name + " (S=" + std::to_string(param) + ")"
            << " time: " << std::setw(8) << std::fixed << std::setprecision(2)
            << seconds << " s"
            << " (" << std::setw(6) << std::fixed << std::setprecision(2)
            << gflops << " GFLOPS)" << std::endl;
}

void testCorrectness(const Matrix &A, const Matrix &B) {
  std::cout << "\nVerifying correctness of all algorithms" << std::endl;

  Matrix C_classic = createMatrix(N);
  multiplyClassic(A, B, C_classic);
  std::cout << "Classic algorithm - OK" << std::endl;

  Matrix C_transpose = createMatrix(N);
  multiplyTranspose(A, B, C_transpose);
  std::cout << "With transpose: "
            << (compareMatrices(C_classic, C_transpose) ? "OK" : "FAIL")
            << std::endl;

  Matrix C_buffered = createMatrix(N);
  multiplyBuffered(A, B, C_buffered);
  std::cout << "With buffering: "
            << (compareMatrices(C_classic, C_buffered) ? "OK" : "FAIL")
            << std::endl;

  Matrix C_block = createMatrix(N);
  multiplyBlock(A, B, C_block, 64);
  std::cout << "Block (S=64): "
            << (compareMatrices(C_classic, C_block) ? "OK" : "FAIL")
            << std::endl;
}

void testBlockSizeVariations(const Matrix &A, const Matrix &B) {
  std::cout << "\nTesting block multiplication with different block sizes"
            << std::endl;
  std::vector<int> blockSizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

  for (int S : blockSizes) {
    if (S > static_cast<int>(N))
      continue;
    Matrix C = createMatrix(N);
    setZero(C);
    auto start = std::chrono::high_resolution_clock::now();
    multiplyBlock(A, B, C, S);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double seconds = duration.count() / 1'000'000.0;
    double gflops = (2.0 * N * N * N) / (seconds * 1e9);

    globalResults.blockSizes.push_back({S, gflops});

    std::cout << std::left << std::setw(35)
              << "Block (S=" + std::to_string(S) + ")"
              << " time: " << std::setw(8) << std::fixed << std::setprecision(2)
              << seconds << " s"
              << " (" << std::setw(6) << std::fixed << std::setprecision(2)
              << gflops << " GFLOPS)" << std::endl;
  }
}

void testLoopUnrolling(const Matrix &A, const Matrix &B) {
  std::cout << "\nTesting loop unrolling (buffered method)" << std::endl;

  Matrix C_base = createMatrix(N);
  auto decorator = make_decorator(multiplyBuffered);
  double base_seconds = decorator(A, B, C_base);
  double base_gflops = (2.0 * N * N * N) / (base_seconds * 1e9);
  globalResults.unrollingFactors.push_back({1, base_gflops});

  std::cout << std::left << std::setw(35) << "Buffered (M=1)"
            << " time: " << std::setw(8) << std::fixed << std::setprecision(2)
            << base_seconds << " s"
            << " (" << std::setw(6) << std::fixed << std::setprecision(2)
            << base_gflops << " GFLOPS)" << std::endl;

  std::vector<int> unrollFactors = {2, 4, 8, 16};

  for (int M : unrollFactors) {
    Matrix C = createMatrix(N);
    setZero(C);
    std::string name = "Buffered (M=" + std::to_string(M) + ")";

    auto start = std::chrono::high_resolution_clock::now();

    switch (M) {
    case 2:
      multiplyBufferedUnrolled<2>(A, B, C);
      break;
    case 4:
      multiplyBufferedUnrolled<4>(A, B, C);
      break;
    case 8:
      multiplyBufferedUnrolled<8>(A, B, C);
      break;
    case 16:
      multiplyBufferedUnrolled<16>(A, B, C);
      break;
    default:
      multiplyBuffered(A, B, C);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double seconds = duration.count() / 1'000'000.0;
    double gflops = (2.0 * N * N * N) / (seconds * 1e9);

    globalResults.unrollingFactors.push_back({M, gflops});

    std::cout << std::left << std::setw(35) << name << " time: " << std::setw(8)
              << std::fixed << std::setprecision(2) << seconds << " s"
              << " (" << std::setw(6) << std::fixed << std::setprecision(2)
              << gflops << " GFLOPS)" << std::endl;
  }
}

void testTransposeImpact(const Matrix &A, const Matrix &B) {
  std::cout << "\nImpact of transpose operation on performance" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  Matrix Bt = transpose(B);
  auto mid = std::chrono::high_resolution_clock::now();

  Matrix C_transpose_only = createMatrix(N);
  multiplyTranspose(A, B, C_transpose_only);
  auto end = std::chrono::high_resolution_clock::now();

  auto transpose_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(mid - start);
  double transpose_seconds = transpose_duration.count() / 1'000'000.0;

  auto multiply_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - mid);
  double multiply_seconds = multiply_duration.count() / 1'000'000.0;

  double total_seconds = transpose_seconds + multiply_seconds;
  double gflops_total = (2.0 * N * N * N) / (total_seconds * 1e9);
  double gflops_multiply = (2.0 * N * N * N) / (multiply_seconds * 1e9);

  globalResults.transposeTime = transpose_seconds;
  globalResults.multiplyTimeTranspose = multiply_seconds;

  std::cout << "Transpose time: " << transpose_seconds << " s" << std::endl;
  std::cout << "Multiplication time (without transpose): " << multiply_seconds
            << " s"
            << " (" << gflops_multiply << " GFLOPS)" << std::endl;
  std::cout << "Total time (with transpose): " << total_seconds << " s"
            << " (" << gflops_total << " GFLOPS)" << std::endl;
}

void testCombinedOptimizations(const Matrix &A, const Matrix &B) {
  std::cout << "\nCombined test (block + loop unrolling)" << std::endl;
  std::vector<int> blockSizes = {32, 64, 128};
  std::vector<int> unrollFactors = {2, 4, 8};

  for (int S : blockSizes) {
    for (int M : unrollFactors) {
      Matrix C = createMatrix(N);
      setZero(C);
      std::string name =
          "Block S=" + std::to_string(S) + " M=" + std::to_string(M);

      auto start = std::chrono::high_resolution_clock::now();

      size_t n = A.size();
      for (size_t ii = 0; ii < n; ii += S) {
        for (size_t jj = 0; jj < n; jj += S) {
          for (size_t kk = 0; kk < n; kk += S) {
            size_t iMax = std::min(ii + S, n);
            size_t jMax = std::min(jj + S, n);
            size_t kMax = std::min(kk + S, n);

            for (size_t i = ii; i < iMax; ++i) {
              for (size_t j = jj; j < jMax; ++j) {
                float s[8] = {0.0f};
                size_t k;

                for (k = kk; k + M - 1 < kMax; k += M) {
                  for (int u = 0; u < M; ++u) {
                    s[u] += A[i][k + u] * B[k + u][j];
                  }
                }

                float total = 0.0f;
                for (int u = 0; u < M; ++u) {
                  total += s[u];
                }
                for (; k < kMax; ++k) {
                  total += A[i][k] * B[k][j];
                }

                C[i][j] += total;
              }
            }
          }
        }
      }

      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      double seconds = duration.count() / 1'000'000.0;
      double gflops = (2.0 * N * N * N) / (seconds * 1e9);

      globalResults.combinedResults.push_back({S, M, gflops});

      std::cout << std::left << std::setw(30) << name
                << " time: " << std::setw(8) << std::fixed
                << std::setprecision(2) << seconds << " s"
                << " (" << std::setw(6) << std::fixed << std::setprecision(2)
                << gflops << " GFLOPS)" << std::endl;
    }
  }
}

void saveToXML() {
  std::ofstream file("results.xml");

  file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
  file << "<performance_results>\n";
  file << "  <matrix_size>" << N << "</matrix_size>\n";
  file << "  <floating_point_operations>" << (2.0 * N * N * N)
       << "</floating_point_operations>\n\n";

  file << "  <basic_algorithms>\n";
  for (const auto &alg : globalResults.basicAlgorithms) {
    file << "    <algorithm name=\"" << alg.first << "\" gflops=\""
         << alg.second << "\" />\n";
  }
  file << "  </basic_algorithms>\n\n";

  file << "  <block_size_results>\n";
  for (const auto &bs : globalResults.blockSizes) {
    file << "    <block size=\"" << bs.first << "\" gflops=\"" << bs.second
         << "\" />\n";
  }
  file << "  </block_size_results>\n\n";

  file << "  <unrolling_results>\n";
  for (const auto &ur : globalResults.unrollingFactors) {
    file << "    <unroll_factor m=\"" << ur.first << "\" gflops=\"" << ur.second
         << "\" />\n";
  }
  file << "  </unrolling_results>\n\n";

  file << "  <transpose_impact>\n";
  file << "    <transpose_time>" << globalResults.transposeTime
       << "</transpose_time>\n";
  file << "    <multiply_time_without_transpose>"
       << globalResults.multiplyTimeTranspose
       << "</multiply_time_without_transpose>\n";
  double classic_gflops = 0;
  for (const auto &alg : globalResults.basicAlgorithms) {
    if (alg.first == "1. Classic")
      classic_gflops = alg.second;
  }
  double transpose_gflops = 0;
  for (const auto &alg : globalResults.basicAlgorithms) {
    if (alg.first == "2. With transpose")
      transpose_gflops = alg.second;
  }
  file << "    <classic_gflops>" << classic_gflops << "</classic_gflops>\n";
  file << "    <transpose_gflops>" << transpose_gflops
       << "</transpose_gflops>\n";
  file << "  </transpose_impact>\n\n";

  file << "  <combined_results>\n";
  for (const auto &comb : globalResults.combinedResults) {
    file << "    <combined block_size=\"" << std::get<0>(comb)
         << "\" unroll_factor=\"" << std::get<1>(comb) << "\" gflops=\""
         << std::get<2>(comb) << "\" />\n";
  }
  file << "  </combined_results>\n";

  file << "</performance_results>\n";
  file.close();

  std::cout << "\nResults saved to results.xml" << std::endl;
}

int main() {
  std::cout << std::fixed << std::setprecision(2);

  std::cout << "Matrix size: " << N << "x" << N << std::endl;
  std::cout << "Computational volume: " << (2.0 * N * N * N / 1e9) << " GFLOPS"
            << std::endl;
  std::cout << "Memory per matrix: "
            << (N * N * sizeof(float) / (1024.0 * 1024.0)) << " MB"
            << std::endl;

  Matrix A = createMatrix(N);
  Matrix B = createMatrix(N);
  fillRandom(A);
  fillRandom(B);

  testCorrectness(A, B);

  std::cout << "\nPerformance comparison of basic algorithms" << std::endl;

  Matrix C_classic = createMatrix(N);
  measureAndPrint("1. Classic", A, B, C_classic, multiplyClassic);

  Matrix C_transpose = createMatrix(N);
  measureAndPrint("2. With transpose", A, B, C_transpose, multiplyTranspose);

  Matrix C_buffered = createMatrix(N);
  measureAndPrint("3. Buffered", A, B, C_buffered, multiplyBuffered);

  Matrix C_block = createMatrix(N);
  setZero(C_block);
  auto start = std::chrono::high_resolution_clock::now();
  multiplyBlock(A, B, C_block, 64);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double seconds = duration.count() / 1'000'000.0;
  double gflops = (2.0 * N * N * N) / (seconds * 1e9);
  std::cout << std::left << std::setw(35) << "4. Block (S=64)"
            << " time: " << std::setw(8) << std::fixed << std::setprecision(2)
            << seconds << " s"
            << " (" << std::setw(6) << std::fixed << std::setprecision(2)
            << gflops << " GFLOPS)" << std::endl;
  globalResults.basicAlgorithms.push_back({"4. Block", gflops});

  testBlockSizeVariations(A, B);
  testLoopUnrolling(A, B);
  testTransposeImpact(A, B);
  testCombinedOptimizations(A, B);

  saveToXML();

  return 0;
}
