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
	float Str[4] = { 100.0f, 95.0f, 90.0f, 85.0f };

	float Tmt[4] = { 3.0f / 12.0f, 6.0f / 12.0f, 1.0f, 2.0f };

	cudaMemcpyToSymbol(sigmad, sigma, 10 * sizeof(float));
	cudaMemcpyToSymbol(thetad, theta, 10 * sizeof(float));
	cudaMemcpyToSymbol(kappad, kappa, 10 * sizeof(float));
	cudaMemcpyToSymbol(Strd, Str, 4 * sizeof(float));

	int pidx, same;
	int NTPB = 32;
	int NB =  125;
	int Ntraj = 40000; 
	float dt = 1.0f / (64.0f * 24.0f);
	float StrR, kappaR, sigmaR, error, price;

	curandState* states;
	cudaMalloc(&states, NB*NTPB*sizeof(curandState));
	init_curand_state_k <<<NB, NTPB>>> (states);
	float *sum;
	int* num;
	cudaMallocManaged(&sum, 2*NB*NTPB*sizeof(float));
	cudaMallocManaged(&num, NB * NTPB * sizeof(int));

	int numTraj;
	FILE* fpt;

	char strg[30];
	for(int i=0; i<4; i++){
		MC_k<<<NB,NTPB>>>(dt, Tmt[i], Ntraj, states, sum, num);
		cudaDeviceSynchronize();
		for(int j=0; j<16; j++){
		//for (int j = 0; j < 6; j++) {
			StrR = Str[j];
			sprintf(strg, "Tmt%.4fStr%.4f.csv", Tmt[i], StrR);
			fpt = fopen(strg, "w+");
			fprintf(fpt, "alpha, beta, m, rho, Y0, price, 95cI, numTraj\n");
			for(int k=0; k< 1296*8; k++){
				same = k + j*(1296 * 8);
				numTraj = num[same];
				pidx = j;
				price = sum[2*same];
				error = 1.96f*sqrtf(sum[2*same+1] - (price * price)) / sqrtf((float)Ntraj);
				same -= (pidx* 1296 * 8);
				pidx = same/(216*8);
				mR = m[pidx];
				same -= (pidx*216*8);
				pidx = same/(216);
				alphaR = alpha[pidx];
				same -= (pidx*216);
				pidx = same/(36);
				betaR = sqrtf(2.0f*alphaR*nu2[pidx])*(1.0f - expf(mR));//beta[pidx];
				same -= (pidx*36);
				pidx = same/(6);
				rhoR = rho[pidx];
				same -= (pidx*6);
				pidx = same;
				fprintf(fpt, "%f, %f, %f, %f, %f, %f, %f, %d\n", alphaR, betaR, mR, rhoR, Y0[same], price, error, numTraj);
			}
			fclose(fpt);
		}
	}

	cudaFree(states);
	cudaFree(sum);
	cudaFree(num);

	return 0;
}