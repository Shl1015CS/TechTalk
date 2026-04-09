#include <stdio.h>
#define CUDA_CHECK(call) do{\
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "CUDA ERROR at  %s: %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}while (0)


__global__ void vector_add(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 10000000;
    size_t bytes = N * sizeof(float);

    // host
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);
    for(int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    //Device
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes,cudaMemcpyHostToDevice));

    //kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    vector_add<<<grid_size,block_size>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    int correct = 1;
    for(int i = 0; i < N; ++i) {
        if( h_c[i] != 3.0f) {
            correct = 0;
            break;
        }
    }
    printf("Result: %s\n", correct ? "PASS" : "FAIL");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}