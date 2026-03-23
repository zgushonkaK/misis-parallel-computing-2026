#include <x86intrin.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

struct CPUIDResult {
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;
};

struct VendorInfo {
    unsigned int max_basic_function;
    std::string vendor_string;
};

struct ProcessorVersionInfo {
    unsigned int stepping;
    unsigned int model;
    unsigned int family;
    unsigned int processor_type;
    unsigned int extended_model;
    unsigned int extended_family;
    unsigned int full_identifier;
};

struct ProcessorFeatures {
    bool fpu;
    bool tsc;
    bool mmx;
    bool sse;
    bool sse2;
    bool htt;
    bool sse3;
    bool ssse3;
    bool fma3;
    bool sse41;
    bool sse42;
    bool avx;
    unsigned int max_logical_processors;
    unsigned int local_apic_id;
};

struct CacheInfo {
    unsigned int cache_type;
    unsigned int cache_level;
    bool fully_associative;
    unsigned int threads_per_cache;
    unsigned int cores_per_cache;
    unsigned int cache_line_size;
    unsigned int physical_line_partitions;
    unsigned int ways_of_associativity;
    unsigned int number_of_sets;
    bool inclusive;
    unsigned int cache_size;
};

struct ExtendedFeatures {
    bool avx2;
    bool rtm;
    bool avx512f;
    bool sha;
    bool gfni;
    bool amx_bf16;
    bool amx_tile;
    bool amx_int8;
};

struct FrequencyInfo {
    unsigned int base_frequency_mhz;
    unsigned int max_frequency_mhz;
    unsigned int bus_frequency_mhz;
};

struct ExtendedVendorInfo {
    unsigned int max_extended_function;
    std::string brand_string;
};

struct ExtendedFeaturesAMD {
    bool sse4a;
    bool fma4;
    bool threednow;
    bool threednow_ext;
};

class CPUIDManager {
private:
    VendorInfo vendor_info;
    ProcessorVersionInfo version_info;
    ProcessorFeatures features;
    ExtendedFeatures extended_features;
    std::vector<CacheInfo> cache_info;
    FrequencyInfo frequency_info;
    ExtendedVendorInfo extended_info;
    ExtendedFeaturesAMD amd_features;

    CPUIDResult cpuid(unsigned int function_id) {
        CPUIDResult result;
        __cpuid(function_id, result.eax, result.ebx, result.ecx, result.edx);
        return result;
    }

    CPUIDResult cpuidex(unsigned int function_id, unsigned int subfunction_id) {
        CPUIDResult result;
        __cpuidex((int*)&result, function_id, subfunction_id);
        return result;
    }

    void fetchVendorInfo() {
        CPUIDResult result = cpuid(0);
        vendor_info.max_basic_function = result.eax;

        char vendor[13] = {0};
        memcpy(vendor + 0, &result.ebx, 4);
        memcpy(vendor + 4, &result.edx, 4);
        memcpy(vendor + 8, &result.ecx, 4);
        vendor_info.vendor_string = std::string(vendor);
    }

    void fetchVersionInfo() {
        CPUIDResult result = cpuid(1);
        unsigned int eax = result.eax;

        version_info.full_identifier = eax;
        version_info.stepping = eax & 0xF;
        version_info.model = (eax >> 4) & 0xF;
        version_info.family = (eax >> 8) & 0xF;
        version_info.processor_type = (eax >> 12) & 0x3;
        version_info.extended_model = (eax >> 16) & 0xF;
        version_info.extended_family = (eax >> 20) & 0xFF;
    }

    void fetchFeatures() {
        CPUIDResult result = cpuid(1);
        unsigned int ebx = result.ebx;
        unsigned int ecx = result.ecx;
        unsigned int edx = result.edx;

        features.fpu = (edx >> 0) & 1;
        features.tsc = (edx >> 4) & 1;
        features.mmx = (edx >> 23) & 1;
        features.sse = (edx >> 25) & 1;
        features.sse2 = (edx >> 26) & 1;
        features.htt = (edx >> 28) & 1;

        features.sse3 = (ecx >> 0) & 1;
        features.ssse3 = (ecx >> 9) & 1;
        features.fma3 = (ecx >> 12) & 1;
        features.sse41 = (ecx >> 19) & 1;
        features.sse42 = (ecx >> 20) & 1;
        features.avx = (ecx >> 28) & 1;

        features.max_logical_processors = (ebx >> 16) & 0xFF;
        features.local_apic_id = ebx & 0xFF;
    }

    void fetchExtendedFeatures() {
        extended_features = {0};

        if (vendor_info.max_basic_function >= 7) {
            CPUIDResult result = cpuidex(7, 0);

            extended_features.avx2 = (result.ebx >> 5) & 1;
            extended_features.rtm = (result.ebx >> 11) & 1;
            extended_features.avx512f = (result.ebx >> 16) & 1;
            extended_features.sha = (result.ebx >> 29) & 1;

            extended_features.gfni = (result.ecx >> 8) & 1;

            extended_features.amx_bf16 = (result.edx >> 22) & 1;
            extended_features.amx_tile = (result.edx >> 24) & 1;
            extended_features.amx_int8 = (result.edx >> 25) & 1;
        }
    }

    void fetchCacheInfo() {
        cache_info.clear();
        unsigned int subleaf = 0;

        while (true) {
            CPUIDResult result = cpuidex(4, subleaf);

            if ((result.eax & 0x1F) == 0) break;

            CacheInfo cache;
            cache.cache_type = result.eax & 0x1F;
            cache.cache_level = (result.eax >> 5) & 0x7;
            cache.fully_associative = (result.eax >> 8) & 1;

            cache.threads_per_cache = ((result.ebx >> 14) & 0xFFF) + 1;
            cache.cores_per_cache = ((result.ebx >> 26) & 0x3F) + 1;

            cache.number_of_sets = result.ecx + 1;

            cache.inclusive = (result.edx >> 2) & 1;
            cache.cache_line_size = (result.ebx & 0xFFF) + 1;
            cache.physical_line_partitions = ((result.ebx >> 12) & 0x3FF) + 1;
            cache.ways_of_associativity = ((result.ebx >> 22) & 0x3FF) + 1;

            cache.cache_size = cache.cache_line_size *
                              cache.physical_line_partitions *
                              cache.ways_of_associativity *
                              cache.number_of_sets;

            cache_info.push_back(cache);
            subleaf++;

            if (subleaf > 20) break;
        }
    }

    void fetchFrequencyInfo() {
        frequency_info = {0};

        if (vendor_info.max_basic_function >= 0x16) {
            CPUIDResult result = cpuid(0x16);
            frequency_info.base_frequency_mhz = result.eax & 0xFFFF;
            frequency_info.max_frequency_mhz = result.ebx & 0xFFFF;
            frequency_info.bus_frequency_mhz = result.ecx & 0xFFFF;
        }
    }

    void fetchExtendedInfo() {
        CPUIDResult result = cpuid(0x80000000);
        extended_info.max_extended_function = result.eax;

        if (extended_info.max_extended_function >= 0x80000004) {
            char brand[49] = {0};
            CPUIDResult results[3];

            for (int i = 0; i < 3; i++) {
                results[i] = cpuid(0x80000002 + i);
                memcpy(brand + i*16, &results[i].eax, 4);
                memcpy(brand + i*16 + 4, &results[i].ebx, 4);
                memcpy(brand + i*16 + 8, &results[i].ecx, 4);
                memcpy(brand + i*16 + 12, &results[i].edx, 4);
            }

            extended_info.brand_string = std::string(brand);
        }
    }

    void fetchAMDFeatures() {
        amd_features = {0};

        if (vendor_info.max_basic_function >= 0x80000001 &&
            vendor_info.vendor_string == "AuthenticAMD") {
            CPUIDResult result = cpuid(0x80000001);

            amd_features.sse4a = (result.ecx >> 6) & 1;
            amd_features.fma4 = (result.ecx >> 16) & 1;

            amd_features.threednow = (result.edx >> 31) & 1;
            amd_features.threednow_ext = (result.edx >> 30) & 1;
        }
    }

    std::string getCacheTypeString(unsigned int type) const {
        switch(type) {
            case 1: return "Data Cache";
            case 2: return "Instruction Cache";
            case 3: return "Unified Cache";
            default: return "Unknown";
        }
    }

public:
    CPUIDManager() {
        fetchAllData();
    }

    void fetchAllData() {
        fetchVendorInfo();
        fetchVersionInfo();
        fetchFeatures();
        fetchExtendedFeatures();
        fetchCacheInfo();
        fetchFrequencyInfo();
        fetchExtendedInfo();
        fetchAMDFeatures();
    }

    const VendorInfo& getVendorInfo() const { return vendor_info; }
    const ProcessorVersionInfo& getVersionInfo() const { return version_info; }
    const ProcessorFeatures& getFeatures() const { return features; }
    const ExtendedFeatures& getExtendedFeatures() const { return extended_features; }
    const std::vector<CacheInfo>& getCacheInfo() const { return cache_info; }
    const FrequencyInfo& getFrequencyInfo() const { return frequency_info; }
    const ExtendedVendorInfo& getExtendedInfo() const { return extended_info; }
    const ExtendedFeaturesAMD& getAMDFeatures() const { return amd_features; }

    void printAllInfo() const {
        printHeader("CPU Information");

        printVendorInfo();
        printProcessorBrand();
        printVersionInfo();
        printFeatures();
        printExtendedFeatures();
        printCacheInfo();
        printFrequencyInfo();
        printAMDInfo();
        printExtendedFunctionsInfo();
    }

    void printVendorInfo() const {
        std::cout << "\n[Vendor Information]" << std::endl;
        std::cout << "  Vendor: " << vendor_info.vendor_string << std::endl;
        std::cout << "  Max Basic Function: 0x" << std::hex
                  << vendor_info.max_basic_function << std::dec << std::endl;
    }

    void printProcessorBrand() const {
        if (!extended_info.brand_string.empty()) {
            std::cout << "\n[Processor Brand]" << std::endl;
            std::cout << "  " << extended_info.brand_string << std::endl;
        }
    }

    void printVersionInfo() const {
        std::cout << "\n[Processor Version]" << std::endl;
        std::cout << "  Family:          " << version_info.family << std::endl;
        std::cout << "  Extended Family: " << version_info.extended_family << std::endl;
        std::cout << "  Model:           " << version_info.model << std::endl;
        std::cout << "  Extended Model:  " << version_info.extended_model << std::endl;
        std::cout << "  Stepping:        " << version_info.stepping << std::endl;
        std::cout << "  Processor Type:  " << version_info.processor_type << std::endl;
        std::cout << "  Full Identifier: 0x" << std::hex
                  << version_info.full_identifier << std::dec << std::endl;
    }

    void printFeatures() const {
        std::cout << "\n[Processor Features]" << std::endl;
        printFeature("FPU", features.fpu);
        printFeature("TSC", features.tsc);
        printFeature("MMX", features.mmx);
        printFeature("SSE", features.sse);
        printFeature("SSE2", features.sse2);
        printFeature("SSE3", features.sse3);
        printFeature("SSSE3", features.ssse3);
        printFeature("SSE4.1", features.sse41);
        printFeature("SSE4.2", features.sse42);
        printFeature("AVX", features.avx);
        printFeature("FMA3", features.fma3);
        printFeature("Hyper-Threading", features.htt);
        std::cout << "  Logical Processors: " << features.max_logical_processors << std::endl;
        std::cout << "  Local APIC ID:      " << features.local_apic_id << std::endl;
    }

    void printExtendedFeatures() const {
        std::cout << "\n[Extended Features]" << std::endl;
        printFeature("AVX2", extended_features.avx2);
        printFeature("AVX-512F", extended_features.avx512f);
        printFeature("SHA", extended_features.sha);
        printFeature("GFNI", extended_features.gfni);
        printFeature("RTM (TSX)", extended_features.rtm);
        printFeature("AMX-BF16", extended_features.amx_bf16);
        printFeature("AMX-TILE", extended_features.amx_tile);
        printFeature("AMX-INT8", extended_features.amx_int8);
    }

    void printCacheInfo() const {
        if (cache_info.empty()) return;

        std::cout << "\n[Cache Information]" << std::endl;
        for (size_t i = 0; i < cache_info.size(); i++) {
            const auto& cache = cache_info[i];
            std::cout << "  L" << cache.cache_level << " "
                      << getCacheTypeString(cache.cache_type) << ":" << std::endl;
            std::cout << "    Size:           " << cache.cache_size / 1024 << " KB"
                      << " (" << cache.cache_size << " bytes)" << std::endl;
            std::cout << "    Associativity:  " << cache.ways_of_associativity << "-way" << std::endl;
            std::cout << "    Line Size:      " << cache.cache_line_size << " bytes" << std::endl;
            std::cout << "    Sets:           " << cache.number_of_sets << std::endl;
            if (cache.threads_per_cache > 1) {
                std::cout << "    Shared by:      " << cache.threads_per_cache << " threads" << std::endl;
            }
            std::cout << "    Inclusive:      " << (cache.inclusive ? "Yes" : "No") << std::endl;
        }
    }

    void printFrequencyInfo() const {
        if (frequency_info.base_frequency_mhz == 0) return;

        std::cout << "\n[Frequencies]" << std::endl;
        std::cout << "  Base Frequency:     " << frequency_info.base_frequency_mhz << " MHz" << std::endl;
        if (frequency_info.max_frequency_mhz > 0) {
            std::cout << "  Max Turbo:          " << frequency_info.max_frequency_mhz << " MHz" << std::endl;
        }
        if (frequency_info.bus_frequency_mhz > 0) {
            std::cout << "  Bus Frequency:      " << frequency_info.bus_frequency_mhz << " MHz" << std::endl;
        }
    }

    void printAMDInfo() const {
        if (vendor_info.vendor_string != "AuthenticAMD") return;

        std::cout << "\n[AMD Specific Features]" << std::endl;
        printFeature("SSE4a", amd_features.sse4a);
        printFeature("FMA4", amd_features.fma4);
        printFeature("3DNow!", amd_features.threednow);
        printFeature("3DNow! Ext", amd_features.threednow_ext);
    }

    void printExtendedFunctionsInfo() const {
        std::cout << "\n[Extended Functions]" << std::endl;
        std::cout << "  Max Extended Function: 0x" << std::hex
                  << extended_info.max_extended_function << std::dec << std::endl;
    }

private:
    void printHeader(const std::string& title) const {
        std::cout << "\n╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║ " << std::left << std::setw(58) << title << "║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    }

    void printFeature(const std::string& name, bool enabled) const {
        std::cout << "  " << std::left << std::setw(20) << name << ": "
                  << (enabled ? "✓" : "✗") << std::endl;
    }
};

int main() {
    CPUIDManager cpuid;
    cpuid.printAllInfo();

    return 0;
}
