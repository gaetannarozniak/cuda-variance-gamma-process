#include <stdio.h>
#include <curand_kernel.h>
#include <iostream>
#include <math.h>
using namespace std;

#define NP 20 // nuber of sigma, theta and kappa values
#define NS 8 // number of strike values

__device__ float sigmad[NP];
__device__ float thetad[NP];
__device__ float kappad[NP];
__device__ float strd[NS];

// initialize the cuda random state for each thread
__global__ void init_curand_state_k(curandState* state)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(0, idx, 0, &state[idx]);
}

// simulate a Gamma(a) variable using Johnk or Best generator (depending on the value of a)
__device__ float gammaRand(float a, curandState *state)
{
    if (a <= 0.0f) {
        return 0.0f;
    }
    if (a < 1.0f) // Johnk's generator
    {
        while(true)
        {
            float U = curand_uniform(state);
            float V = curand_uniform(state);
            float X = powf(U, 1.0f / a);
            float Y = powf(V, 1.0f / (1.0f - a));
            if (X + Y <= 1.0f)
            {
                float E = -__logf(curand_uniform(state));
                float G = (X / (X + Y)) * E;
                return G;
            }
        }
    }
    else // Best's generator
    {
        float a_minus1 = a - 1.0f;
        float c_       = 3.0f * a - 0.75f;
        while(true)
        {
            float U = curand_uniform(state);
            float V = curand_uniform(state);
            float W = U * (1.0f - U);
            float Y = __fsqrt_rn(c_ / W) * (U - 0.5f);
            float X = a_minus1 + Y;
            if (X >= 0.0f)
            {
                float Z = 64.0f * W * W * W * V * V * V;
                float twoY2overX = 2.0f * Y * Y / X;
                if (Z <= (1.0f - twoY2overX))
                {
                    return X;
                }
                if (__logf(Z) <= 2.0f * (a_minus1 * __logf(X / a_minus1) - Y))
                {
                    return X;
                }
            }
        }
    }
}

// kernel that computes an MC estimate of the price of a call option 
__global__ void MC_VG(float dt, float T, int Ntraj, curandState* state, float* sums, float* sums2) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[idx];

    float str    = strd[idx % NS];
    float kappa  = kappad[(idx / NS) % NP];
    float theta  = thetad[(idx / (NS * NP)) % NP];
    float sigma  = sigmad[(idx / (NS * NP * NP)) % NP];

    float wVG = __logf(1.0f - theta * kappa - kappa * sigma * sigma / 2.0f) / kappa;
    float shape = dt / kappa;
    int Nsteps = (int)(T / dt);

    float payoffSum  = 0.0f;
    float payoffSum2 = 0.0f;

    for(int i = 0; i < Ntraj; i++)
    {
        float X = 0.0f;
        for(int n = 0; n < Nsteps; n++)
        {
            float dS = gammaRand(shape, &localState);
            float N  = curand_normal(&localState);
            float dX = sigma * N * __fsqrt_rn(kappa * dS) + theta * kappa * dS;
            X += dX;
        }
        float Y = __expf(wVG * T + X);
        float payoff = fmaxf(Y - str, 0.0f);
        payoffSum  += payoff;
        payoffSum2 += payoff * payoff;
    }

    sums[idx] = payoffSum;
    sums2[idx] = payoffSum2;
    state[idx] = localState;
}

int main(void) {

    // set the values for the grid of parameters
    float sigma[NP] = {
        0.100f, 0.105f, 0.110f, 0.115f, 0.120f,
        0.125f, 0.130f, 0.135f, 0.140f, 0.145f,
        0.150f, 0.155f, 0.160f, 0.165f, 0.170f,
        0.175f, 0.180f, 0.185f, 0.190f, 0.195f
    };
    float theta[NP] = {
        -0.300f, -0.295f, -0.290f, -0.285f, -0.280f,
        -0.275f, -0.270f, -0.265f, -0.260f, -0.255f,
        -0.250f, -0.245f, -0.240f, -0.235f, -0.230f,
        -0.225f, -0.220f, -0.215f, -0.210f, -0.205f
    };
    float kappa[NP] = {
        0.100f, 0.105f, 0.110f, 0.115f, 0.120f,
        0.125f, 0.130f, 0.135f, 0.140f, 0.145f,
        0.150f, 0.155f, 0.160f, 0.165f, 0.170f,
        0.175f, 0.180f, 0.185f, 0.190f, 0.195f
    };
    // strike values
    float str[NS] = {1.15f, 1.10f, 1.05f, 1.00f, 0.95f, 0.90f, 0.85f, 0.80f};

    // copy the parameter values to device
    cudaMemcpyToSymbol(sigmad, sigma, NP * sizeof(float));
    cudaMemcpyToSymbol(thetad, theta, NP * sizeof(float));
    cudaMemcpyToSymbol(kappad, kappa, NP * sizeof(float));
    cudaMemcpyToSymbol(strd, str, NS * sizeof(float));

    // maturity values (stay on the host)
    float Tmt[5] = {
        3.0f / 12.0f, 6.0f / 12.0f, 9.0f / 12.0f, 1.0f,
        1.5f
    };
    
    int NTPB = 512;
    int NB = 64000 / NTPB;  
    int Ntraj = 10000;
    float dt = 1.0f / (64.0f * 24.0f);

    float strR, kappaR, sigmaR, thetaR, expected_payoff, stdd, error;
    
    curandState* states;
    cudaMalloc(&states, NB * NTPB * sizeof(curandState));
    init_curand_state_k<<<NB, NTPB>>>(states);
    
    float *sum;
    float *sum2;
    cudaMallocManaged(&sum, NB * NTPB * sizeof(float));
    cudaMallocManaged(&sum2, NB * NTPB * sizeof(float));

    FILE* fpt;
    char filename[30];
    sprintf(filename, "training.csv");
    fpt = fopen(filename, "w+");
    fprintf(fpt, "sigma,theta,kappa,strike,T,expected_payoff,error,Ntraj\n");

    int numT = 5;
    int totalComb = NS * NP * NP * NP;

    for(int i = 0; i < numT; i++){
        // call the MC_VG kernel that computes the MC estimates of the prices
        MC_VG<<<NB, NTPB>>>(dt, Tmt[i], Ntraj, states, sum, sum2);
        cudaDeviceSynchronize();
        for(int k = 0; k < totalComb; k++){
            expected_payoff = sum[k] / Ntraj;
            stdd = sqrt(sum2[k] / Ntraj - expected_payoff * expected_payoff);
            error = 1.96 * stdd / sqrt(Ntraj);
            strR    = str[k % NS];
            kappaR  = kappa[(k / NS) % NP];
            thetaR  = theta[(k / (NS * NP)) % NP];
            sigmaR  = sigma[(k / (NS * NP * NP)) % NP];
            fprintf(fpt, "%f, %f, %f, %f, %f, %f, %f, %d\n", sigmaR, thetaR, kappaR, strR, Tmt[i], expected_payoff, error, Ntraj);
        }
    }
    fclose(fpt);
    cudaFree(states);
    cudaFree(sum);
    cudaFree(sum2);
    return 0;
}