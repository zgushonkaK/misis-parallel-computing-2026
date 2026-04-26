// =====================================================================
// Laboratory work #7. Variant 8.
// Box blur 4x4 operator:  d = (1/16) * H,
// where H = sum of pixel values in a 4x4 window.
//
// Target: Apple Silicon (ARM64) with NEON.
// =====================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <arm_neon.h>

static inline std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

#pragma pack(push, 1)
struct BmpFileHeader {
    std::uint16_t bfType;
    std::uint32_t bfSize;
    std::uint16_t bfReserved1;
    std::uint16_t bfReserved2;
    std::uint32_t bfOffBits;
};
struct BmpInfoHeader {
    std::uint32_t biSize;
    std::int32_t  biWidth;
    std::int32_t  biHeight;
    std::uint16_t biPlanes;
    std::uint16_t biBitCount;
    std::uint32_t biCompression;
    std::uint32_t biSizeImage;
    std::int32_t  biXPelsPerMeter;
    std::int32_t  biYPelsPerMeter;
    std::uint32_t biClrUsed;
    std::uint32_t biClrImportant;
};
#pragma pack(pop)

bool load_bmp_as_grayscale(const std::string& path,
                           std::vector<std::uint8_t>& gray, int& W, int& H)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "Cannot open file: %s\n", path.c_str()); return false; }

    BmpFileHeader fh{};
    BmpInfoHeader ih{};
    f.read(reinterpret_cast<char*>(&fh), sizeof(fh));
    f.read(reinterpret_cast<char*>(&ih), sizeof(ih));
    if (!f || fh.bfType != 0x4D42) { std::fprintf(stderr, "Not a BMP or corrupted.\n"); return false; }
    if (ih.biBitCount != 24 || ih.biCompression != 0) {
        std::fprintf(stderr,
            "Only 24-bit uncompressed BMP is supported. "
            "File has biBitCount=%u, biCompression=%u\n",
            ih.biBitCount, ih.biCompression);
        return false;
    }

    W = ih.biWidth;
    H = std::abs(ih.biHeight);
    bool topDown = ih.biHeight < 0;
    int rowBytes = ((W * 3 + 3) / 4) * 4;

    f.seekg(fh.bfOffBits, std::ios::beg);
    std::vector<std::uint8_t> row(rowBytes);
    gray.assign(static_cast<std::size_t>(W) * H, 0);

    for (int y = 0; y < H; ++y) {
        f.read(reinterpret_cast<char*>(row.data()), rowBytes);
        if (!f) { std::fprintf(stderr, "Error reading pixels.\n"); return false; }
        int dstY = topDown ? y : (H - 1 - y);
        std::uint8_t* out = gray.data() + dstY * W;
        for (int x = 0; x < W; ++x) {
            std::uint8_t B = row[x * 3 + 0];
            std::uint8_t G = row[x * 3 + 1];
            std::uint8_t R = row[x * 3 + 2];
            int Y = (77 * R + 150 * G + 29 * B + 128) >> 8;
            if (Y > 255) Y = 255;
            out[x] = static_cast<std::uint8_t>(Y);
        }
    }
    return true;
}

bool save_bmp_grayscale(const std::string& path,
                        const std::vector<std::uint8_t>& gray, int W, int H)
{
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "Cannot create file: %s\n", path.c_str()); return false; }

    int rowBytes = ((W + 3) / 4) * 4;
    std::uint32_t paletteSize = 256 * 4;
    std::uint32_t pixelOffset = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader) + paletteSize;

    BmpFileHeader fh{};
    fh.bfType    = 0x4D42;
    fh.bfSize    = pixelOffset + rowBytes * H;
    fh.bfOffBits = pixelOffset;

    BmpInfoHeader ih{};
    ih.biSize         = sizeof(BmpInfoHeader);
    ih.biWidth        = W;
    ih.biHeight       = H;
    ih.biPlanes       = 1;
    ih.biBitCount     = 8;
    ih.biCompression  = 0;
    ih.biSizeImage    = rowBytes * H;
    ih.biClrUsed      = 256;
    ih.biClrImportant = 256;

    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    f.write(reinterpret_cast<const char*>(&ih), sizeof(ih));

    for (int i = 0; i < 256; ++i) {
        std::uint8_t entry[4] = {
            static_cast<std::uint8_t>(i),
            static_cast<std::uint8_t>(i),
            static_cast<std::uint8_t>(i),
            0
        };
        f.write(reinterpret_cast<const char*>(entry), 4);
    }

    std::vector<std::uint8_t> row(rowBytes, 0);
    for (int y = H - 1; y >= 0; --y) {
        std::memcpy(row.data(), gray.data() + y * W, W);
        f.write(reinterpret_cast<const char*>(row.data()), rowBytes);
    }
    return static_cast<bool>(f);
}

// ---------------------------------------------------------------------
// SCALAR reference implementation.
//
// IMPORTANT: modern Clang on Apple Silicon is very aggressive about
// auto-vectorizing simple loops into NEON. To keep this function a
// truly scalar baseline (which is what the lab requires — comparing
// hand-written SIMD against a high-level non-SIMD implementation),
// we explicitly disable optimization for this function.
// Without this pragma, the "scalar" version is already auto-vectorized
// and comparison with the NEON version becomes meaningless.
// ---------------------------------------------------------------------
#pragma clang optimize off
void blur4x4_scalar(const std::uint8_t* src, std::uint8_t* dst, int W, int H)
{
    std::memset(dst, 0, static_cast<std::size_t>(W) * H);
    for (int y = 0; y <= H - 4; ++y) {
        for (int x = 0; x <= W - 4; ++x) {
            std::uint16_t sum = 0;
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 4; ++i)
                    sum += src[(y + j) * W + (x + i)];
            dst[y * W + x] = static_cast<std::uint8_t>(sum >> 4);   // / 16
        }
    }
}
#pragma clang optimize on

// ---------------------------------------------------------------------
// NEON implementation. Processes 8 pixels per iteration.
// ---------------------------------------------------------------------
static inline uint16x8_t load8_u8_to_u16(const std::uint8_t* p)
{
    uint8x8_t v = vld1_u8(p);
    return vmovl_u8(v);
}

static inline uint16x8_t hsum4_u16(uint16x8_t cur, uint16x8_t nxt)
{
    uint16x8_t s1 = vextq_u16(cur, nxt, 1);
    uint16x8_t s2 = vextq_u16(cur, nxt, 2);
    uint16x8_t s3 = vextq_u16(cur, nxt, 3);
    uint16x8_t sum = vaddq_u16(cur, s1);
    sum            = vaddq_u16(sum, s2);
    sum            = vaddq_u16(sum, s3);
    return sum;
}

void blur4x4_neon(const std::uint8_t* src, std::uint8_t* dst, int W, int H)
{
    std::memset(dst, 0, static_cast<std::size_t>(W) * H);

    const int xLimit = W - 16;

    for (int y = 0; y <= H - 4; ++y) {
        const std::uint8_t* row0 = src + (y + 0) * W;
        const std::uint8_t* row1 = src + (y + 1) * W;
        const std::uint8_t* row2 = src + (y + 2) * W;
        const std::uint8_t* row3 = src + (y + 3) * W;
        std::uint8_t*       out  = dst + y * W;

        int x = 0;
        for (; x <= xLimit; x += 8) {
            uint16x8_t r0a = load8_u8_to_u16(row0 + x);
            uint16x8_t r0b = load8_u8_to_u16(row0 + x + 8);
            uint16x8_t r1a = load8_u8_to_u16(row1 + x);
            uint16x8_t r1b = load8_u8_to_u16(row1 + x + 8);
            uint16x8_t r2a = load8_u8_to_u16(row2 + x);
            uint16x8_t r2b = load8_u8_to_u16(row2 + x + 8);
            uint16x8_t r3a = load8_u8_to_u16(row3 + x);
            uint16x8_t r3b = load8_u8_to_u16(row3 + x + 8);

            uint16x8_t h0 = hsum4_u16(r0a, r0b);
            uint16x8_t h1 = hsum4_u16(r1a, r1b);
            uint16x8_t h2 = hsum4_u16(r2a, r2b);
            uint16x8_t h3 = hsum4_u16(r3a, r3b);

            uint16x8_t H_ = vaddq_u16(vaddq_u16(h0, h1),
                                     vaddq_u16(h2, h3));

            H_ = vshrq_n_u16(H_, 4);

            uint8x8_t packed = vqmovn_u16(H_);
            vst1_u8(out + x, packed);
        }

        for (; x <= W - 4; ++x) {
            std::uint16_t sum = 0;
            for (int j = 0; j < 4; ++j)
                for (int i = 0; i < 4; ++i)
                    sum += src[(y + j) * W + (x + i)];
            out[x] = static_cast<std::uint8_t>(sum >> 4);
        }
    }
}

// ---------------------------------------------------------------------
// Benchmark helpers.
// ---------------------------------------------------------------------
struct BenchStats {
    double min_ns;
    double median_ns;
    double mean_ns;
};

template <typename F>
BenchStats benchmark(F&& fn, int repeats)
{
    std::vector<std::int64_t> samples;
    samples.reserve(repeats);
    for (int r = 0; r < repeats; ++r) {
        std::int64_t t0 = now_ns();
        fn();
        std::int64_t t1 = now_ns();
        samples.push_back(t1 - t0);
    }
    std::vector<std::int64_t> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    double med = (sorted.size() % 2 == 1)
        ? static_cast<double>(sorted[sorted.size() / 2])
        : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    std::int64_t sum = 0;
    for (auto v : samples) sum += v;
    double mean = static_cast<double>(sum) / samples.size();
    return BenchStats{ static_cast<double>(sorted.front()), med, mean };
}

int main(int argc, char** argv)
{
    std::cout << "SIMD backend: NEON (Apple Silicon / ARM64)\n";

    if (argc < 2) {
        std::printf("Usage: %s <input.bmp> [--repeats N]\n", argv[0]);
        std::printf("  Input:  24-bit BMP (color, uncompressed).\n");
        std::printf("  Output: <input>_gray.bmp, <input>_blur_scalar.bmp, <input>_blur_neon.bmp\n");
        return 1;
    }

    std::string inputPath = argv[1];
    int REPEATS = 100;
    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--repeats" && i + 1 < argc) {
            REPEATS = std::max(1, std::atoi(argv[++i]));
        }
    }

    std::vector<std::uint8_t> src;
    int W = 0, H = 0;
    if (!load_bmp_as_grayscale(inputPath, src, W, H)) return 1;
    std::printf("Loaded: %s   (%d x %d, %d pixels)\n",
                inputPath.c_str(), W, H, W * H);
    if (W < 16 || H < 4) { std::fprintf(stderr, "Image is too small.\n"); return 1; }

    std::string stem = inputPath;
    std::size_t dot = stem.find_last_of('.');
    if (dot != std::string::npos) stem = stem.substr(0, dot);

    save_bmp_grayscale(stem + "_gray.bmp", src, W, H);

    std::vector<std::uint8_t> dstScalar(static_cast<std::size_t>(W) * H, 0);
    std::vector<std::uint8_t> dstNEON  (static_cast<std::size_t>(W) * H, 0);

    // Warm-up (cache, branch predictor).
    blur4x4_scalar(src.data(), dstScalar.data(), W, H);
    blur4x4_neon  (src.data(), dstNEON  .data(), W, H);

    BenchStats sc = benchmark(
        [&]{ blur4x4_scalar(src.data(), dstScalar.data(), W, H); }, REPEATS);
    BenchStats nn = benchmark(
        [&]{ blur4x4_neon  (src.data(), dstNEON  .data(), W, H); }, REPEATS);

    int firstDiff = -1;
    for (std::size_t i = 0; i < dstScalar.size(); ++i)
        if (dstScalar[i] != dstNEON[i]) { firstDiff = static_cast<int>(i); break; }

    std::printf("\nResult comparison: %s\n",
                firstDiff < 0 ? "MATCH (implementations are equivalent)"
                              : "MISMATCH");
    if (firstDiff >= 0) {
        int y = firstDiff / W, x = firstDiff % W;
        std::printf("  first difference at (%d, %d): scalar=%u, neon=%u\n",
                    x, y, dstScalar[firstDiff], dstNEON[firstDiff]);
    }

    const double NS_PER_MS = 1.0e6;
    const double NS_PER_US = 1.0e3;
    const double pixels    = static_cast<double>(W) * static_cast<double>(H);

    auto print_row = [&](const char* label, const BenchStats& s) {
        std::printf("  %-18s  min %10.4f ms  |  median %10.4f ms  |  mean %10.4f ms\n",
                    label, s.min_ns / NS_PER_MS, s.median_ns / NS_PER_MS, s.mean_ns / NS_PER_MS);
        std::printf("  %-18s  min %10.3f us  |  median %10.3f us  |  mean %10.3f us\n",
                    "",    s.min_ns / NS_PER_US, s.median_ns / NS_PER_US, s.mean_ns / NS_PER_US);
        std::printf("  %-18s  min %10.3f ns/px | median %10.3f ns/px | mean %10.3f ns/px\n",
                    "",    s.min_ns / pixels,   s.median_ns / pixels,   s.mean_ns / pixels);
    };

    std::printf("\nTiming (averaged over %d runs):\n", REPEATS);
    print_row("C++ scalar",  sc);
    std::printf("\n");
    print_row("NEON",        nn);

    std::printf("\nSpeedup (scalar / NEON):\n");
    std::printf("  by min:     x%.3f\n", sc.min_ns    / nn.min_ns);
    std::printf("  by median:  x%.3f\n", sc.median_ns / nn.median_ns);
    std::printf("  by mean:    x%.3f\n", sc.mean_ns   / nn.mean_ns);

    save_bmp_grayscale(stem + "_blur_scalar.bmp", dstScalar, W, H);
    save_bmp_grayscale(stem + "_blur_neon.bmp",   dstNEON,   W, H);
    std::printf("\nSaved:\n  %s_gray.bmp\n  %s_blur_scalar.bmp\n  %s_blur_neon.bmp\n",
                stem.c_str(), stem.c_str(), stem.c_str());
    return firstDiff < 0 ? 0 : 1;
}
