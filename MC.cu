/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>

__device__ float Y0d[6];
__device__ float md[6];
__device__ float alphad[8];
__device__ float nu2d[6];
__device__ float rhod[6];
//__device__ float Strd[6];
__device__ float Strd[16];


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// Set the state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

// Monte Carlo simulation kernel
__global__ void MC_k(float dt, float T, int Ntraj, curandState* state, float* sum, int* num){

	int pidx, same, numR;
	float t, X, Y;
	int idx = blockDim.x * blockIdx.x + threadIdx.x; 
	curandState localState = state[idx];
	float2 G;
	float B;
	float price;
	float sumR = 0.0f;
	float sum2R = 0.0f;
	same = idx;

	pidx = same/(1296 * 8);
	float StrR = Strd[pidx];
	same -= (pidx* 1296 * 8);
	pidx = same/(216 * 8);
	float mR = md[pidx];
	same -= (pidx* 216 * 8);
	pidx = same/(216);
	float alphaR = alphad[pidx];
	same -= (pidx*216);
	pidx = same/(36);
	float betaR = sqrtf(2.0f*alphaR*nu2d[pidx])*(1.0f - expf(mR));//betad[pidx];
	same -= (pidx*36);
	pidx = same/(6);
	float rhoR = rhod[pidx];
	same -= (pidx*6);
	pidx = same;

	numR = 0;
	for (int i = 0; i < Ntraj; i++) {
		t = 0.0f;
		X = 1.0f;
		Y = Y0d[pidx];
		while(t<T){
			G = curand_normal2(&localState);
			X *= (1.0f + expf(Y)*G.x*dt);
			B = rhoR*G.x + sqrtf(1.0f-rhoR*rhoR)*G.y;
			Y = Y + alphaR*(mR-Y)*dt*dt + betaR*dt*B;
			t += dt*dt;
		}
		if (X < 12.0f) {
			price = fmaxf(0.0f, X - StrR) / Ntraj;
			sumR += price;
			sum2R += price * price * Ntraj;
			numR++;
		}
	}
	sum[2*idx] = sumR*((float)Ntraj/numR);
	sum[2*idx + 1] = sum2R*((float)Ntraj / numR);
	num[idx] = numR;


	/* Copy state back to global memory */
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