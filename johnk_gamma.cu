#include <cstdio>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath>

/**
 * Kernel to initialize cuRAND states
 */
__global__ void setupStates(curandState *states, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread gets same seed, a sequence number = idx, no offset
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Kernel that generates n Gamma(a, 1) r.v.s by Johnk’s method (shape a <= 1).
 * d_out[tid] will hold one Gamma deviate per thread.
 */
__global__ void generateGammaJohnk(curandState *states, float *d_out, int n, float a)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    curandState localState = states[tid];

    float X, Y, U, V, E;
    while (true) {
        // 1) Generate two uniforms
        U = curand_uniform(&localState);
        V = curand_uniform(&localState);

        // 2) Raise them to powers (Johnk’s step)
        X = powf(U, 1.0f / a);
        Y = powf(V, 1.0f / (1.0f - a));

        // 3) Check rejection condition
        if (X + Y <= 1.0f) {
            // 4) Generate an exponential
            E = -logf(curand_uniform(&localState));
            // 5) Output final gamma deviate
            d_out[tid] = (X / (X + Y)) * E;
            break;
        }
    }

    // Store back the updated RNG state
    states[tid] = localState;
}

int main(int argc, char **argv)
{
    // --- 1) Parse command-line
    if (argc < 3) {
        printf("Usage: %s <N> <shape a in (0,1]> [seed]\n", argv[0]);
        return 1;
    }
    int N          = atoi(argv[1]);    // number of gamma variables
    float a        = atof(argv[2]);    // shape <= 1
    unsigned long seed = (argc > 3) ? strtoul(argv[3], NULL, 10) : 1234UL;

    if (a <= 0.0f || a > 1.0f) {
        printf("Shape parameter 'a' must be in (0, 1].\n");
        return 1;
    }

    // --- 2) Allocate device memory
    float *d_out;
    curandState *d_states;

    cudaMalloc(&d_out,    N * sizeof(float));
    cudaMalloc(&d_states, N * sizeof(curandState));

    // --- 3) Launch setup kernel
    int blockSize = 128;
    int gridSize  = (N + blockSize - 1) / blockSize;
    setupStates<<<gridSize, blockSize>>>(d_states, seed);
    cudaDeviceSynchronize();

    // --- 4) Generate Gamma(a,1) variates
    generateGammaJohnk<<<gridSize, blockSize>>>(d_states, d_out, N, a);
    cudaDeviceSynchronize();

    // --- 5) Copy results back to host
    float *h_out = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // --- 6) Print first few
    int toPrint = (N < 10) ? N : 10;
    printf("First %d Gamma(%g,1) samples:\n", toPrint, a);
    for (int i = 0; i < toPrint; i++) {
        printf("%f\n", h_out[i]);
    }

    // --- 7) Cleanup
    free(h_out);
    cudaFree(d_out);
    cudaFree(d_states);
    return 0;
}
