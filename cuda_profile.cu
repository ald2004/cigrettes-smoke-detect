#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime_api.h>
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
//#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
//#endif
    return result;
}
static cudaStream_t streamsArray[16];    // cudaStreamSynchronize( get_cuda_stream() );
static int streamInit[16] = { 0 };
void profileCopies(float* h_a,
    float* h_b,
    float* d,
    unsigned int  n,
    char* desc)
{
    printf("\n%s transfers\n", desc);

    unsigned int bytes = n * sizeof(float);

    // events for timing
    cudaEvent_t startEvent, stopEvent;

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    float time;
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
    printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    checkCuda(cudaEventRecord(startEvent, 0));
    checkCuda(cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
    printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

    for (int i = 0; i < n; ++i) {
        if (h_a[i] != h_b[i]) {
            printf("*** %s transfers failed ***\n", desc);
            break;
        }
    }

    // clean up events
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
}
int cuda_get_device()
{
    int n = 0;
    cudaError_t status = cudaGetDevice(&n);
    checkCuda(status);
    return n;
}
cudaStream_t get_cuda_stream() {
    int i = cuda_get_device();
    if (!streamInit[i]) {
        //printf("Create CUDA-stream \n");
        cudaError_t status = cudaStreamCreate(&streamsArray[i]);
        //cudaError_t status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamNonBlocking);
        if (status != cudaSuccess) {
            printf(" cudaStreamCreate error: %d \n", status);
            const char* s = cudaGetErrorString(status);
            printf("CUDA Error: %s\n", s);
            status = cudaStreamCreateWithFlags(&streamsArray[i], cudaStreamDefault);
            checkCuda(status);
        }
        streamInit[i] = 1;
    }
    return streamsArray[i];
}


void error(const char* s)
{
    perror(s);
    assert(0);
    exit(EXIT_FAILURE);
}
float* cuda_make_array(float* x, size_t n)
{
    float* x_gpu;
    size_t size = sizeof(float) * n;
    cudaError_t status = cudaMalloc((void**)&x_gpu, size);
    //cudaError_t status = cudaMallocManaged((void **)&x_gpu, size, cudaMemAttachGlobal);
    //status = cudaMemAdvise(x_gpu, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    if (status != cudaSuccess) fprintf(stderr, " Try to set subdivisions=64 in your cfg-file. \n");
    checkCuda(status);
    if (x) {
        //status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
        status = cudaMemcpyAsync(x_gpu, x, size, cudaMemcpyDefault, get_cuda_stream());
        checkCuda(status);
    }
    if (!x_gpu) error("Cuda malloc failed\n");
    return x_gpu;
}
int main()
{
    int n;
    cudaDeviceProp prop;
    
    unsigned int nElements = 4 * 1024 * 1024;
    const unsigned int bytes = nElements * sizeof(float);

    
    cudaError_t status;
    int a[] = { 0, 1 };
    for (auto i : a) {
        checkCuda(cudaGetDeviceProperties(&prop, i));
        printf("\nDevice: %s\n", prop.name);
        printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));
        status = cudaSetDevice(i);
        checkCuda(status);
        status = cudaGetDevice(&n);
        checkCuda(status);
        status = cudaStreamCreate(&streamsArray[i]);
        checkCuda(status);
        status = cudaStreamCreate(&streamsArray[i]);
        checkCuda(status);

        float* xx=cuda_make_array(0, 4435968);
        printf("%f ", xx);
        
    }
    getchar();
    return 0;
}
