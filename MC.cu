#include <stdio.h>
#include <curand_kernel.h>

__device__ float sigmad[10];
__device__ float thetad[10];
__device__ float kappad[10];
__device__ float strd[4];


// Set the state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}


__device__ float gammaRand(float a, float b, curandState *state)
{
    // Handle invalid shape gracefully:
    if (a <= 0.0f) {
        // Could return 0, or assert, etc. 
        // We'll just return 0 here to avoid NaNs.
        return 0.0f;
    }

    if (a < 1.0f)
    {
        // We will produce Gamma(a, 1) and then scale by 1/b to get Gamma(a, b).
        // Johnk’s accept–reject:
        while(true)
        {
            float U = curand_uniform(state);
            float V = curand_uniform(state);

            // X ~ U^(1/a),  Y ~ V^(1/(1-a))
            float X = powf(U, 1.0f / a);
            float Y = powf(V, 1.0f / (1.0f - a));

            if (X + Y <= 1.0f)
            {
                // Once accepted, generate an exponential(1) deviate E = -ln(U2)
                float E = -__logf(curand_uniform(state));
                // This yields a Gamma(a,1) sample = (X/(X+Y)) * E
                float G = (X / (X + Y)) * E;  
                // Finally convert to Gamma(a, b) => scale by 1/b
                return G * (1.0f / b);
            }
        }
    }
    else
    {
        // Also known as the “Cheng–Best” or “BC” method
        // We'll generate Gamma(a,1) and then multiply by 1/b => Gamma(a,b).

        float a_minus1 = a - 1.0f;
        float c_       = 3.0f * a - 0.75f;  

        while(true)
        {
            float U = curand_uniform(state);
            float V = curand_uniform(state);

            // W = U(1-U).  Because 0 < U < 1, W in (0, 0.25].
            float W = U * (1.0f - U);
            // Y ~ +/- sqrt( c_/W ) * (U - 0.5). 
            // This is the “stretch” transform.
            float Y = __fsqrt_rn(c_ / W) * (U - 0.5f);
            // X ~ (a-1) + Y
            float X = a_minus1 + Y;

            // Accept only if X >= 0.
            if (X >= 0.0f)
            {
                // Z = 64 W^3 V^3
                float Z = 64.0f * W * W * W * V * V * V;
                // Now do the acceptance checks:
                // 1) quick reject if Z > (1 - 2Y^2 / X)
                float twoY2overX = 2.0f * Y * Y / X; 
                if (Z <= (1.0f - twoY2overX))
                {
                    // This is a direct accept => X is Gamma(a,1)
                    return X * (1.0f / b);
                }
                // 2) otherwise, do the log check:
                if (__logf(Z) <= 2.0f * (a_minus1 * __logf(X / a_minus1) - Y))
                {
                    // accepted => X is Gamma(a,1)
                    return X * (1.0f / b);
                }
            }
            // else reject and repeat
        }
    }
}

__global__ void MC_VG(
	float dt,
    float T,       
    int   Ntraj,
    curandState* state, 
    float* sums
)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[idx];

	// unpack parameters
	float str = strd[idx % 4];
	float kappa = kappad[(idx / 4) % 10];
	float theta = thetad[(idx / 40) % 10];
	float sigma = sigmad[(idx / 400) % 10];

	float wVG = __logf(1.0f - theta * kappa - kappa * sigma * sigma / 2.0f) / kappa;

    float shape = dt / kappa;
    float scale = kappa;
	int Nsteps = (int)(T / dt);

    float payoffSum  = 0.0f;  // accumulate payoff

    // Main loop over # of paths assigned to this thread
    for(int i = 0; i < Ntraj; i++)
    {
        // We will accumulate X_t from 0..T, then exponentiate
        float X = 0.0f;

        // Simulate the partial sums for X(t) from t=0..T
        for(int n = 0; n < Nsteps; n++)
        {
            // 1) Sample Gamma increment:
            float dS = gammaRand(shape, scale, &localState);

            // 2) Sample Normal(0,1):
            float N  = curand_normal(&localState);

            // 3) Variance Gamma increment:
            float dX = sigma * N * __fsqrt_rn(kappa * dS)
                       + theta * dS;
            X += dX;
        }

        // final asset price at T:  Y_T = exp( wVG*T + X )
        float Y = __expf(wVG * T + X);

        // payoff of a call option:
        float payoff = fmaxf(Y - K, 0.0f);

        payoffSum  += payoff;
    }

    // store partial sums (later you can reduce across threads)
    sums[idx] = payoffSum;    // \sum of payoffs

    // save state
    state[idx] = localState;
}


int main(void) {
	float sigma[10] = { 0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.2f };
	float theta[10] = { -0.34f, -0.3f, -0.27f, -0.24f, -0.21f, -0.18f, -0.15f, -0.12f, -0.09f, -0.06f };
	float kappa[10] = { 0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.2f };
	float str[4] = { 100.0f, 95.0f, 90.0f, 85.0f };

	float Tmt[4] = { 3.0f / 12.0f, 6.0f / 12.0f, 1.0f, 2.0f };

	cudaMemcpyToSymbol(sigmad, sigma, 10 * sizeof(float));
	cudaMemcpyToSymbol(thetad, theta, 10 * sizeof(float));
	cudaMemcpyToSymbol(kappad, kappa, 10 * sizeof(float));
	cudaMemcpyToSymbol(strd, str, 4 * sizeof(float));

	int pidx, same;
	int NTPB = 32;
	int NB =  125;
	int Ntraj = 40000; 
	float dt = 1.0f / (64.0f * 24.0f);
	float strR, kappaR, sigmaR, thetaR, expected_payoff;

	curandState* states;
	cudaMalloc(&states, NB*NTPB*sizeof(curandState));
	init_curand_state_k <<<NB, NTPB>>> (states);
	float *sum;
	cudaMallocManaged(&sum, NB*NTPB*sizeof(float));
	FILE* fpt;

	char strg[30];
	for(int i=0; i<4; i++){
		MC_k<<<NB,NTPB>>>(dt, Tmt[i], Ntraj, states, sum);
		cudaDeviceSynchronize();
		sprintf(strg, "Tmt%.4f.csv", Tmt[i]);
		fpt = fopen(strg, "w+");
		fprintf(fpt, "sigma, theta, kappa, Str, expected_payoff, Ntraj\n");
		for(int k=0; k< 10 * 10 * 10 * 4; k++){
			expected_payoff = sum[k] / Ntraj;
			strR = str[k % 4];
			kappaR = kappa[(k / 4) % 10];
			thetaR = theta[(k / 40) % 10];
			sigmaR = sigma[(k / 400) % 10];
			fprintf(fpt, "%f, %f, %f, %f, %f, %f, %f, %d\n", sigmaR, thetaR, kappaR, strR, expected_payoff, Ntraj);
		}
		fclose(fpt);
	}
	cudaFree(states);
	cudaFree(sum);
	cudaFree(num);

	return 0;
}