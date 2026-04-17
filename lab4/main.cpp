#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>

class XMLWriter {
private:
    std::ofstream file;
    int indent;

    void writeIndent() {
        for (int i = 0; i < indent; ++i) {
            file << "  ";
        }
    }

public:
    XMLWriter(const std::string& filename) : indent(0) {
        file.open(filename);
        file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    }

    ~XMLWriter() {
        if (file.is_open()) {
            file.close();
        }
    }

    void openElement(const std::string& name, const std::string& attributes = "") {
        writeIndent();
        file << "<" << name;
        if (!attributes.empty()) {
            file << " " << attributes;
        }
        file << ">\n";
        indent++;
    }

    void closeElement(const std::string& name) {
        indent--;
        writeIndent();
        file << "</" << name << ">\n";
    }

    void writeElement(const std::string& name, const std::string& value, const std::string& attributes = "") {
        writeIndent();
        file << "<" << name;
        if (!attributes.empty()) {
            file << " " << attributes;
        }
        file << ">" << value << "</" << name << ">\n";
    }

    void writeValueElement(const std::string& name, double value, const std::string& unit = "") {
        std::string attrs;
        if (!unit.empty()) {
            attrs = "unit=\"" + unit + "\"";
        }
        writeElement(name, std::to_string(value), attrs);
    }
};

template <class... Args>
struct ExeTime {
public:
    ExeTime(std::function<void(Args...)> func) : f_(func) {}

    double operator()(Args... args) {
        auto start = std::chrono::high_resolution_clock::now();
        f_(args...);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return duration.count() / 1'000'000'000.0;
    }

private:
    std::function<void(Args...)> f_;
};

template <class... Args>
ExeTime<Args...> make_decorator(void (*f)(Args...)) {
    return ExeTime<Args...>(std::function<void(Args...)>(f));
}

void sequentialSum(const std::vector<int>& arr, volatile long long& result) {
    result = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        result += arr[i];
    }
}

void randomSumOnFly(const std::vector<int>& arr, volatile long long& result) {
    static thread_local std::mt19937 rng(std::random_device{}());
    result = 0;
    size_t n = arr.size();

    for (size_t i = 0; i < n; ++i) {
        size_t idx = rng() % n;
        result += arr[idx];
    }
}

void randomSumPreIndexed(const std::vector<int>& arr, const std::vector<size_t>& indices, volatile long long& result) {
    result = 0;
    for (size_t i = 0; i < arr.size(); ++i) {
        result += arr[indices[i]];
    }
}

void runSequential(const std::vector<int>& arr) {
    volatile long long result;
    sequentialSum(arr, result);
}

void runRandomOnFly(const std::vector<int>& arr) {
    volatile long long result;
    randomSumOnFly(arr, result);
}

struct MeasurementResult {
    size_t size_bytes;
    double sequential_time_ns;
    double random_onfly_time_ns;
    double random_preindexed_time_ns;
    int repeats;
};

std::vector<size_t> generateSizes() {
    std::vector<size_t> sizes;

    for (size_t kb = 1; kb <= 2048; kb += 64) {
        sizes.push_back(kb * 1024);
    }

    for (size_t mb = 2; mb <= 32; mb += 1) {
        sizes.push_back(mb * 1024 * 1024);
    }

    for (size_t mb = 35; mb <= 150; mb += 5) {
        sizes.push_back(mb * 1024 * 1024);
    }

    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    return sizes;
}

void saveResultsToXML(const std::vector<MeasurementResult>& results, const std::string& filename) {
    XMLWriter writer(filename);

    writer.openElement("cache_latency_measurements");

    writer.openElement("system_info");
    writer.writeElement("cpu_model", "Unknown"); // You can add CPU detection
    writer.writeElement("timestamp", std::to_string(std::time(nullptr)));
    writer.closeElement("system_info");

    writer.openElement("results");

    for (const auto& r : results) {
        writer.openElement("measurement", "size_bytes=\"" + std::to_string(r.size_bytes) + "\"");
        writer.writeValueElement("size_kb", r.size_bytes / 1024.0, "KB");
        writer.writeValueElement("sequential_time_ns", r.sequential_time_ns, "ns");
        writer.writeValueElement("random_onfly_time_ns", r.random_onfly_time_ns, "ns");
        writer.writeValueElement("random_preindexed_time_ns", r.random_preindexed_time_ns, "ns");
        writer.writeValueElement("repeats", r.repeats, "");
        writer.closeElement("measurement");
    }

    writer.closeElement("results");
    writer.closeElement("cache_latency_measurements");
}

double measureTime(const std::function<void()>& func, int warmup = 1) {
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return duration.count();
}

MeasurementResult measureForSize(size_t size_bytes, int repeats = 5) {
    size_t num_elements = size_bytes / sizeof(int);
    MeasurementResult result;
    result.size_bytes = size_bytes;
    result.repeats = repeats;

    std::cout << "  Measuring size: " << size_bytes / 1024 << " KB ("
              << num_elements << " elements)..." << std::endl;

    std::vector<int> arr(num_elements, 1); // Fill with 1s
    std::vector<size_t> indices(num_elements);

    std::mt19937 rng(std::random_device{}());
    for (size_t i = 0; i < num_elements; ++i) {
        indices[i] = rng() % num_elements;
    }

    double total_seq = 0;
    for (int r = 0; r < repeats; ++r) {
        total_seq += measureTime([&]() {
            volatile long long res;
            sequentialSum(arr, res);
        }, 0);
    }
    result.sequential_time_ns = total_seq / (repeats * num_elements);

    double total_rand_onfly = 0;
    for (int r = 0; r < repeats; ++r) {
        total_rand_onfly += measureTime([&]() {
            volatile long long res;
            randomSumOnFly(arr, res);
        }, 0);
    }
    result.random_onfly_time_ns = total_rand_onfly / (repeats * num_elements);

    double total_rand_pre = 0;
    for (int r = 0; r < repeats; ++r) {
        total_rand_pre += measureTime([&]() {
            volatile long long res;
            randomSumPreIndexed(arr, indices, res);
        }, 0);
    }
    result.random_preindexed_time_ns = total_rand_pre / (repeats * num_elements);

    return result;
}

int main() {
    std::cout << "=== Cache Latency Measurement Program ===" << std::endl;
    std::cout << "Measuring L1, L2, L3 cache and RAM latencies" << std::endl;
    std::cout << std::endl;

    std::cout << "System Information:" << std::endl;
    #ifdef __linux__
    system("lscpu | grep -E 'Model name|CPU MHz|L1d cache|L1i cache|L2 cache|L3 cache' || true");
    #elif defined(_WIN32)
    std::cout << "  (Run on Windows - CPU info not auto-detected)" << std::endl;
    #endif
    std::cout << std::endl;

    std::vector<size_t> sizes = generateSizes();
    std::vector<MeasurementResult> results;

    std::cout << "Will measure " << sizes.size() << " different sizes" << std::endl;
    std::cout << "From " << sizes.front() / 1024 << " KB to "
              << sizes.back() / (1024 * 1024) << " MB" << std::endl;
    std::cout << std::endl;

    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << "Progress: " << i + 1 << "/" << sizes.size() << std::endl;
        MeasurementResult r = measureForSize(sizes[i], 3); // 3 repeats
        results.push_back(r);

        std::cout << "    Sequential: " << std::fixed << std::setprecision(2)
                  << r.sequential_time_ns << " ns/iter" << std::endl;
        std::cout << "    Random on-fly: " << r.random_onfly_time_ns << " ns/iter" << std::endl;
        std::cout << "    Random pre-indexed: " << r.random_preindexed_time_ns << " ns/iter" << std::endl;
        std::cout << std::endl;
    }

    std::string filename = "cache_latency_results.xml";
    saveResultsToXML(results, filename);
    std::cout << "Results saved to " << filename << std::endl;

    return 0;
}
