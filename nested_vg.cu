#include <cstdio>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>


/*
 * DEVICE FUNCTION: gammaRand
 * Generates one Gamma(shape=a, rate=b) random deviate on the GPU.
 *    - If a <= 1, uses Johnk’s method (Algorithm 6.7).
 *    - If a >= 1, uses Best’s method (Algorithm 6.8).
 * You might split them out or use a library; here we inline both for demo.
 */
__device__ float gammaRand(float a, float b, curandState *state)
{
    // We want Gamma(a, b), i.e. shape=a, rate=b.
    // shape a>0, b>0.  The mean is a/b, variance a/(b^2).

    if (a <= 0.f) {
        // Fallback: invalid shape => return 0 or assert
        return 0.f;
    }

    // For a <= 1, use Johnk’s (Algorithm 6.7)
    if (a < 1.0f) {
        while(true) {
            float U = curand_uniform(state);
            float V = curand_uniform(state);

            float X = powf(U, 1.f/a);
            float Y = powf(V, 1.f/(1.f - a));

            if (X + Y <= 1.f) {
                // generate E ~ Exp(1)
                float E = -logf(curand_uniform(state));
                // final gamma deviate has shape a, rate=1, then scale=1/b => multiply by (1/b)
                float G = (X / (X + Y)) * E;  // this is Gamma(a, 1)
                return G * (1.f / b);        // convert to Gamma(a, b)
            }
        }
    }
    else {
        // For a >= 1, use Best’s method (Algorithm 6.8)
        float b_ = a - 1.f;
        float c_ = 3.f*a - 0.75f;
        while(true) {
            float U = curand_uniform(state);
            float V = curand_uniform(state);

            float W = U*(1.f - U);
            float Y = sqrtf(c_/W)*(U - 0.5f);
            float X = b_ + Y;

            if (X >= 0.f) {
                float Z = 64.f*W*W*W * V*V*V;
                // acceptance check
                if ((Z <= (1.f - 2.f*Y*Y/X)) || (logf(Z) <= 2.f*(b_*logf(X/b_) - Y))) {
                    // gamma(a,1)
                    float G = X;
                    return G * (1.f / b);  // scale by 1/b => gamma(a, b)
                }
            }
        }
    }
}

/*
 * KERNEL: setup PRNG states for each thread
 */
__global__ void setupStates(curandState *states, unsigned long seed)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // Each thread gets same seed + its own sequence
    curand_init(seed, tid, 0, &states[tid]);
}

/*
 * KERNEL: Each thread simulates one "outer" path of the Variance Gamma
 *         model on [0, T], with nSteps uniform time steps.
 *         Then computes the payoff (Y_T - K)+ and stores it in d_payoffs[tid].
 *
 *         If "nested" is required, you might do an inner loop inside for
 *         each path, e.g. re-valuing at sub-nodes. This is a minimal example.
 */
__global__ void simulateVGPaths(
    curandState *states,
    float *d_payoffs,
    int nSim, 
    int nSteps,
    float T, 
    float K, 
    float Y0,
    float theta,
    float sigma,
    float kappa
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nSim) return;

    // local copy of RNG state
    curandState localState = states[tid];

    // Basic parameters
    float dt  = T / nSteps; 
    // The drift adjustment w to make VG a martingale under risk-neutral measure
    // One common formula: w = (1 - theta*kappa - 0.5*sigma^2*kappa) / kappa
    // or something similar from your reference.
    float w   = (1.f - theta*kappa - 0.5f*sigma*sigma*kappa) / kappa;

    // Accumulate X(t). Start from 0.
    float X = 0.f;

    for(int i = 0; i < nSteps; i++) {
        // We want ∆S_i ~ Gamma(dt/kappa, 1/kappa) or similar.  Check your exact parameterization:
        // "dt/kappa" is the shape, "1/kappa" is the rate => mean dt
        float shape = dt / kappa;
        float rate  = 1.f / kappa;   // i.e. Gamma(shape, rate)

        float dSi   = gammaRand(shape, rate, &localState);  // ∆S_i
        // Then sample a normal
        float Ni    = curand_normal(&localState);
        // ∆X_i = sigma * sqrt(∆S_i) * Ni + theta * ∆S_i
        float dXi   = sigma * sqrtf(dSi) * Ni + theta * dSi;
        X          += dXi;
    }

    // Final log-price
    float logY = w*T + X;
    // Price
    float YT    = Y0 * expf(logY);
    // Payoff
    float payoff = fmaxf(YT - K, 0.f);

    d_payoffs[tid] = payoff;

    // store back updated RNG state
    states[tid] = localState;
}

/*
 * Host helper: reduce the array of payoffs to get the average (MC estimate).
 * For big N, you'd do a parallel reduction on the GPU, but here's a simple CPU version.
 */
float cpuMean(const float* arr, int n)
{
    double sum = 0.0;
    for(int i=0; i<n; i++){
        sum += arr[i];
    }
    return (float)(sum / n);
}

int main(int argc, char **argv)
{
    // Parse command line: e.g. ./nested_vg 1000000 365 1.0f 100.0f 100.0f 0.1f 0.2f 0.5f 1234
    if (argc < 10) {
        printf("Usage: %s <nSim> <nSteps> <T> <K> <Y0> <theta> <sigma> <kappa> <seed>\n", argv[0]);
        return 1;
    }
    int   nSim   = atoi(argv[1]);
    int   nSteps = atoi(argv[2]);
    float T      = atof(argv[3]);
    float K      = atof(argv[4]);
    float Y0     = atof(argv[5]);
    float theta  = atof(argv[6]);
    float sigma  = atof(argv[7]);
    float kappa  = atof(argv[8]);
    unsigned long seed = strtoul(argv[9], NULL, 10);

    // --- Allocate device arrays
    float *d_payoffs;
    curandState *d_states;

    cudaMalloc(&d_payoffs, nSim * sizeof(float));
    cudaMalloc(&d_states,  nSim * sizeof(curandState));

    // --- Setup RNG states
    int blockSize = 128;
    int gridSize  = (nSim + blockSize - 1) / blockSize;
    setupStates<<<gridSize, blockSize>>>(d_states, seed);
    cudaDeviceSynchronize();

    // --- Launch the simulation kernel
    simulateVGPaths<<<gridSize, blockSize>>>(
        d_states, d_payoffs,
        nSim, nSteps,
        T, K, Y0,
        theta, sigma, kappa
    );
    cudaDeviceSynchronize();

    // --- Copy results back to host
    float *h_payoffs = (float*)malloc(nSim*sizeof(float));
    cudaMemcpy(h_payoffs, d_payoffs, nSim*sizeof(float), cudaMemcpyDeviceToHost);

    // --- Compute mean payoff (option price)
    float call_price = cpuMean(h_payoffs, nSim);

    // Print result
    printf("Estimated call price under Log-VG: %.6f\n", call_price);

    // Cleanup
    free(h_payoffs);
    cudaFree(d_payoffs);
    cudaFree(d_states);
    return 0;
}
