#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <unistd.h>
#include <sys/time.h>
#include <x86intrin.h>
#include <sched.h>
#include <pthread.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cstdint>

uint64_t getMicrotime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

uint64_t getNanotime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

uint64_t rdtsc() {
    return __rdtsc();
}

void setRealtimePriority() {
    sched_param sch_params;
    sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sch_params)) {
        std::cerr << "Failed to set realtime priority" << std::endl;
    }
}

void setAffinity(int coreId) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset)) {
        std::cerr << "Failed to set CPU affinity" << std::endl;
    }
}

void sieveOfEratosthenes(int n) {
    std::vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;

    for (int i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }
}

class MeasurementData {
private:
    std::vector<double> gettimeMeasurements;
    std::vector<double> clockMeasurements;
    std::vector<double> tscMeasurements;
    double cpuFreqGHz;
    int arraySize;
    int numMeasurements;

public:
    MeasurementData(int n, int k) :
        gettimeMeasurements(),
        clockMeasurements(),
        tscMeasurements(),
        cpuFreqGHz(0.0),
        arraySize(n),
        numMeasurements(k) {}


    void setCpuFreq(double freq) {
        cpuFreqGHz = freq;
    }

    void addGettimeMeasurement(double value) {
        gettimeMeasurements.push_back(value);
    }

    void addClockMeasurement(double value) {
        clockMeasurements.push_back(value);
    }

    void addTscMeasurement(double value) {
        tscMeasurements.push_back(value);
    }

    void saveToXML(const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << " for writing" << std::endl;
            return;
        }

        file << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
        file << "<measurements>\n";

        file << "  <system>\n";
        file << "    <cpu_freq_ghz>" << std::fixed << std::setprecision(3) << cpuFreqGHz << "</cpu_freq_ghz>\n";
        file << "    <array_size>" << arraySize << "</array_size>\n";
        file << "    <num_measurements>" << numMeasurements << "</num_measurements>\n";
        file << "  </system>\n";

        file << "  <method name=\"gettimeofday\" unit=\"ms\">\n";
        for (size_t i = 0; i < gettimeMeasurements.size(); ++i) {
            file << "    <measurement index=\"" << i+1 << "\">"
                 << std::fixed << std::setprecision(6) << gettimeMeasurements[i]
                 << "</measurement>\n";
        }
        file << "  </method>\n";

        file << "  <method name=\"clock_gettime\" unit=\"ms\">\n";
        for (size_t i = 0; i < clockMeasurements.size(); ++i) {
            file << "    <measurement index=\"" << i+1 << "\">"
                 << std::fixed << std::setprecision(6) << clockMeasurements[i]
                 << "</measurement>\n";
        }
        file << "  </method>\n";

        file << "  <method name=\"rdtsc\" unit=\"ms\">\n";
        for (size_t i = 0; i < tscMeasurements.size(); ++i) {
            file << "    <measurement index=\"" << i+1 << "\">"
                 << std::fixed << std::setprecision(6) << tscMeasurements[i]
                 << "</measurement>\n";
        }
        file << "  </method>\n";

        file << "</measurements>\n";
        file.close();

        std::cout << "\nResults saved to " << filename << std::endl;
    }
};

int main() {
    setRealtimePriority();
    setAffinity(0);

    const int K = 1000;
    const int N = 50000;

    MeasurementData data(N, K);

    std::cout << "\nAlgorithm: Sieve of Eratosthenes (N = " << N << ")" << std::endl;
    std::cout << "Number of measurements: " << K << "\n" << std::endl;

    sieveOfEratosthenes(N);

    std::cout << "1. Measurement using gettimeofday():" << std::endl;

    for (int i = 0; i < K; ++i) {
        uint64_t start = getMicrotime();

        sieveOfEratosthenes(N);

        uint64_t end = getMicrotime();
        double elapsedMs = (end - start) / 1000.0;

        data.addGettimeMeasurement(elapsedMs);
        std::cout << "  Measurement " << i+1 << ": " << std::fixed << std::setprecision(3)
                  << elapsedMs << " ms" << std::endl;
    }

    std::cout << "\n2. Measurement using clock_gettime():" << std::endl;

    for (int i = 0; i < K; ++i) {
        uint64_t start = getNanotime();

        sieveOfEratosthenes(N);

        uint64_t end = getNanotime();
        double elapsedMs = (end - start) / 1000000.0;

        data.addClockMeasurement(elapsedMs);
        std::cout << "  Measurement " << i+1 << ": " << std::fixed << std::setprecision(3)
                  << elapsedMs << " ms" << std::endl;
    }

    uint64_t tscStart = rdtsc();
    usleep(1000000);
    uint64_t tscEnd = rdtsc();
    double cpuFreqGHz = (tscEnd - tscStart) / 1e9;
    data.setCpuFreq(cpuFreqGHz);

    std::cout << "\n3. Measurement using RDTSC (CPU Freq: " << std::fixed << std::setprecision(2)
              << cpuFreqGHz << " GHz):" << std::endl;

    for (int i = 0; i < K; ++i) {
        uint64_t start = rdtsc();

        sieveOfEratosthenes(N);

        uint64_t end = rdtsc();
        double elapsedMs = (end - start) / (cpuFreqGHz * 1e6); // конвертируем в миллисекунды

        data.addTscMeasurement(elapsedMs);
        std::cout << "  Measurement " << i+1 << ": " << std::fixed << std::setprecision(3)
                  << elapsedMs << " ms (" << (end-start) << " ticks)" << std::endl;
    }

    data.saveToXML("measurements.xml");

    return 0;
}
