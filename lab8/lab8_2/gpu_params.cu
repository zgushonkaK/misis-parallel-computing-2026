#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

int main()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    cout << "CUDA device count: " << device_count << "\n\n";

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, i);

        cout << "===== Device " << i << " =====\n";
        cout << "Name: " << dp.name << "\n\n";

        cout << "Total global memory: "
             << (dp.totalGlobalMem / (1024 * 1024)) << " MB\n";
        cout << "Total const memory: " << dp.totalConstMem << " B\n";
        cout << "Shared memory per block: " << dp.sharedMemPerBlock << " B\n";
        cout << "Registers per block: " << dp.regsPerBlock << "\n\n";

        cout << "WARP size: " << dp.warpSize << "\n";
        cout << "Max threads per block: " << dp.maxThreadsPerBlock << "\n";
        cout << "Max block dimensions: "
             << dp.maxThreadsDim[0] << " x "
             << dp.maxThreadsDim[1] << " x "
             << dp.maxThreadsDim[2] << "\n";
        cout << "Max grid dimensions:  "
             << dp.maxGridSize[0] << " x "
             << dp.maxGridSize[1] << " x "
             << dp.maxGridSize[2] << "\n\n";

        cout << "Compute capability: " << dp.major << "." << dp.minor << "\n";
        cout << "SM (multiprocessor) count: " << dp.multiProcessorCount << "\n";
        cout << "Asynchronous engines count: " << dp.asyncEngineCount << "\n";
        cout << "Copying and computing in parallel: "
             << (dp.deviceOverlap ? 1 : 0) << "\n\n";

        cout << "Core clock: " << (dp.clockRate / 1000) << " MHz\n";
        cout << "Memory clock: " << (dp.memoryClockRate / 1000) << " MHz\n";
        cout << "Memory bus width: " << dp.memoryBusWidth << " bits\n";
        cout << "L2 cache size: " << (dp.l2CacheSize / 1024) << " KB\n";
        cout << "\n";
    }

    return 0;
}
