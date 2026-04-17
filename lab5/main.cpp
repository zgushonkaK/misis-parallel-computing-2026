#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static constexpr size_t N = 1'000'000;
static constexpr size_t N_ALIGNED = (N + 127) & ~size_t(127);

static constexpr int WARMUP_RUNS = 3;
static constexpr int BENCH_RUNS = 51;

template <typename T> static T *aligned_alloc_arr(size_t count) {
  void *p = nullptr;
  if (posix_memalign(&p, 64, count * sizeof(T)) != 0) {
    std::fprintf(stderr, "posix_memalign failed\n");
    std::exit(1);
  }
  return static_cast<T *>(p);
}

using clock_t_ = std::chrono::steady_clock;
using ns_t = std::chrono::nanoseconds;

static inline double ns_to_ms(double ns) { return ns / 1.0e6; }

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double mad(const std::vector<double> &v) {
  double m = median(v);
  std::vector<double> dev(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    dev[i] = std::fabs(v[i] - m);
  return median(std::move(dev));
}

__attribute__((noinline)) static void mul_scalar(const int8_t *__restrict a,
                                                 const int8_t *__restrict b,
                                                 int16_t *__restrict c,
                                                 size_t n) {
  for (size_t i = 0; i < n; ++i) {
    c[i] = static_cast<int16_t>(a[i]) * static_cast<int16_t>(b[i]);
  }
}

__attribute__((noinline)) static void mul_sse2(const int8_t *__restrict a,
                                               const int8_t *__restrict b,
                                               int16_t *__restrict c,
                                               size_t n) {
  const __m128i zero = _mm_setzero_si128();
  for (size_t i = 0; i < n; i += 16) {
    __m128i va = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i));

    __m128i sa = _mm_cmpgt_epi8(zero, va);
    __m128i sb = _mm_cmpgt_epi8(zero, vb);
    __m128i a_lo = _mm_unpacklo_epi8(va, sa);
    __m128i a_hi = _mm_unpackhi_epi8(va, sa);
    __m128i b_lo = _mm_unpacklo_epi8(vb, sb);
    __m128i b_hi = _mm_unpackhi_epi8(vb, sb);

    __m128i r_lo = _mm_mullo_epi16(a_lo, b_lo);
    __m128i r_hi = _mm_mullo_epi16(a_hi, b_hi);

    _mm_store_si128(reinterpret_cast<__m128i *>(c + i), r_lo);
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 8), r_hi);
  }
}

__attribute__((noinline)) static void
mul_sse2_unroll2(const int8_t *__restrict a, const int8_t *__restrict b,
                 int16_t *__restrict c, size_t n) {
  const __m128i zero = _mm_setzero_si128();
  for (size_t i = 0; i < n; i += 32) {
    __m128i va0 = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i va1 = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i + 16));
    __m128i vb0 = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i vb1 = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i + 16));

    __m128i sa0 = _mm_cmpgt_epi8(zero, va0);
    __m128i sa1 = _mm_cmpgt_epi8(zero, va1);
    __m128i sb0 = _mm_cmpgt_epi8(zero, vb0);
    __m128i sb1 = _mm_cmpgt_epi8(zero, vb1);

    __m128i r00 = _mm_mullo_epi16(_mm_unpacklo_epi8(va0, sa0),
                                  _mm_unpacklo_epi8(vb0, sb0));
    __m128i r01 = _mm_mullo_epi16(_mm_unpackhi_epi8(va0, sa0),
                                  _mm_unpackhi_epi8(vb0, sb0));
    __m128i r10 = _mm_mullo_epi16(_mm_unpacklo_epi8(va1, sa1),
                                  _mm_unpacklo_epi8(vb1, sb1));
    __m128i r11 = _mm_mullo_epi16(_mm_unpackhi_epi8(va1, sa1),
                                  _mm_unpackhi_epi8(vb1, sb1));

    _mm_store_si128(reinterpret_cast<__m128i *>(c + i), r00);
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 8), r01);
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 16), r10);
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 24), r11);
  }
}

__attribute__((noinline)) static void
mul_sse2_unroll4(const int8_t *__restrict a, const int8_t *__restrict b,
                 int16_t *__restrict c, size_t n) {
  const __m128i zero = _mm_setzero_si128();
  for (size_t i = 0; i < n; i += 64) {
    __m128i va0 = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i));
    __m128i va1 = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i + 16));
    __m128i va2 = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i + 32));
    __m128i va3 = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i + 48));
    __m128i vb0 = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i));
    __m128i vb1 = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i + 16));
    __m128i vb2 = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i + 32));
    __m128i vb3 = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i + 48));

    __m128i sa0 = _mm_cmpgt_epi8(zero, va0);
    __m128i sa1 = _mm_cmpgt_epi8(zero, va1);
    __m128i sa2 = _mm_cmpgt_epi8(zero, va2);
    __m128i sa3 = _mm_cmpgt_epi8(zero, va3);
    __m128i sb0 = _mm_cmpgt_epi8(zero, vb0);
    __m128i sb1 = _mm_cmpgt_epi8(zero, vb1);
    __m128i sb2 = _mm_cmpgt_epi8(zero, vb2);
    __m128i sb3 = _mm_cmpgt_epi8(zero, vb3);

    _mm_store_si128(reinterpret_cast<__m128i *>(c + i),
                    _mm_mullo_epi16(_mm_unpacklo_epi8(va0, sa0),
                                    _mm_unpacklo_epi8(vb0, sb0)));
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 8),
                    _mm_mullo_epi16(_mm_unpackhi_epi8(va0, sa0),
                                    _mm_unpackhi_epi8(vb0, sb0)));
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 16),
                    _mm_mullo_epi16(_mm_unpacklo_epi8(va1, sa1),
                                    _mm_unpacklo_epi8(vb1, sb1)));
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 24),
                    _mm_mullo_epi16(_mm_unpackhi_epi8(va1, sa1),
                                    _mm_unpackhi_epi8(vb1, sb1)));
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 32),
                    _mm_mullo_epi16(_mm_unpacklo_epi8(va2, sa2),
                                    _mm_unpacklo_epi8(vb2, sb2)));
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 40),
                    _mm_mullo_epi16(_mm_unpackhi_epi8(va2, sa2),
                                    _mm_unpackhi_epi8(vb2, sb2)));
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 48),
                    _mm_mullo_epi16(_mm_unpacklo_epi8(va3, sa3),
                                    _mm_unpacklo_epi8(vb3, sb3)));
    _mm_store_si128(reinterpret_cast<__m128i *>(c + i + 56),
                    _mm_mullo_epi16(_mm_unpackhi_epi8(va3, sa3),
                                    _mm_unpackhi_epi8(vb3, sb3)));
  }
}

__attribute__((noinline)) static void
mul_sse2_unroll8(const int8_t *__restrict a, const int8_t *__restrict b,
                 int16_t *__restrict c, size_t n) {
  const __m128i zero = _mm_setzero_si128();
  for (size_t i = 0; i < n; i += 128) {
    for (size_t k = 0; k < 128; k += 16) {
      __m128i va = _mm_load_si128(reinterpret_cast<const __m128i *>(a + i + k));
      __m128i vb = _mm_load_si128(reinterpret_cast<const __m128i *>(b + i + k));
      __m128i sa = _mm_cmpgt_epi8(zero, va);
      __m128i sb = _mm_cmpgt_epi8(zero, vb);
      _mm_store_si128(reinterpret_cast<__m128i *>(c + i + k),
                      _mm_mullo_epi16(_mm_unpacklo_epi8(va, sa),
                                      _mm_unpacklo_epi8(vb, sb)));
      _mm_store_si128(reinterpret_cast<__m128i *>(c + i + k + 8),
                      _mm_mullo_epi16(_mm_unpackhi_epi8(va, sa),
                                      _mm_unpackhi_epi8(vb, sb)));
    }
  }
}

static void init_data(int8_t *a, int8_t *b, size_t n) {
  std::mt19937 rng(0xC0FFEE);
  std::uniform_int_distribution<int> dist(-128, 127);
  for (size_t i = 0; i < n; ++i) {
    a[i] = static_cast<int8_t>(dist(rng));
    b[i] = static_cast<int8_t>(dist(rng));
  }
}

static bool arrays_equal(const int16_t *x, const int16_t *y, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (x[i] != y[i]) {
      std::fprintf(stderr, "MISMATCH at i=%zu: %d vs %d\n", i, (int)x[i],
                   (int)y[i]);
      return false;
    }
  }
  return true;
}

template <typename Fn>
static void bench(const char *name, Fn &&fn, const int8_t *a, const int8_t *b,
                  int16_t *c, const int16_t *reference, size_t n,
                  std::vector<std::pair<std::string, double>> &results) {
  for (int i = 0; i < WARMUP_RUNS; ++i) {
    std::memset(c, 0, n * sizeof(int16_t));
    fn(a, b, c, n);
  }
  if (reference && !arrays_equal(c, reference, n)) {
    std::fprintf(stderr, "[%s] RESULT MISMATCH!\n", name);
    std::exit(2);
  }

  std::vector<double> times_ns;
  times_ns.reserve(BENCH_RUNS);
  for (int r = 0; r < BENCH_RUNS; ++r) {
    auto t0 = clock_t_::now();
    fn(a, b, c, n);
    auto t1 = clock_t_::now();
    times_ns.push_back(
        (double)std::chrono::duration_cast<ns_t>(t1 - t0).count());
    asm volatile("" : : "r"(c) : "memory");
  }

  double med = median(times_ns);
  double err = mad(times_ns);
  double mn = *std::min_element(times_ns.begin(), times_ns.end());
  double mx = *std::max_element(times_ns.begin(), times_ns.end());

  std::printf("  %-28s  median=%9.3f ms  ±%6.3f  min=%9.3f  max=%9.3f\n", name,
              ns_to_ms(med), ns_to_ms(err), ns_to_ms(mn), ns_to_ms(mx));

  results.emplace_back(name, med);
}

int main() {
  std::printf("=== Лабораторная работа №5. Вариант 8 ===\n");
  std::printf("Тип данных: int8 → int16 (результат)\n");
  std::printf("Операция:   поэлементное умножение  c[i] = a[i] * b[i]\n");
  std::printf("Расширение: SSE2\n");
  std::printf("N = %zu (выровнено до %zu)\n", N, N_ALIGNED);
  std::printf("Прогонов на замер: %d (берётся медиана)\n\n", BENCH_RUNS);

  int8_t *a = aligned_alloc_arr<int8_t>(N_ALIGNED);
  int8_t *b = aligned_alloc_arr<int8_t>(N_ALIGNED);
  int16_t *c = aligned_alloc_arr<int16_t>(N_ALIGNED);
  int16_t *reference = aligned_alloc_arr<int16_t>(N_ALIGNED);

  init_data(a, b, N_ALIGNED);

  std::memset(reference, 0, N_ALIGNED * sizeof(int16_t));
  mul_scalar(a, b, reference, N_ALIGNED);

  std::vector<std::pair<std::string, double>> results;

  std::printf("--- Результаты (время одного прохода по массиву) ---\n");

  bench("1) Scalar C++", mul_scalar, a, b, c, reference, N_ALIGNED, results);
  bench("2) SSE2 (no unroll)", mul_sse2, a, b, c, reference, N_ALIGNED,
        results);
  bench("3) SSE2 unroll x2", mul_sse2_unroll2, a, b, c, reference, N_ALIGNED,
        results);
  bench("4) SSE2 unroll x4", mul_sse2_unroll4, a, b, c, reference, N_ALIGNED,
        results);
  bench("5) SSE2 unroll x8", mul_sse2_unroll8, a, b, c, reference, N_ALIGNED,
        results);

  std::printf("\n--- Ускорение относительно скалярной реализации ---\n");
  double base = results.front().second;
  for (auto &r : results) {
    std::printf("  %-28s  %7.3f ms   speedup = %5.2fx\n", r.first.c_str(),
                ns_to_ms(r.second), base / r.second);
  }

  std::printf("\n--- Пропускная способность (Gelem/s) ---\n");
  for (auto &r : results) {
    double gelem_s = (double)N_ALIGNED / r.second; // elem / ns = Gelem/s
    std::printf("  %-28s  %6.3f Gelem/s\n", r.first.c_str(), gelem_s);
  }

  std::free(a);
  std::free(b);
  std::free(c);
  std::free(reference);
  return 0;
}
