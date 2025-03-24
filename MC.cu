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

void strikeInterval(float* K, float T) {

		float fidx = T * 12.0f + 1.0f;
		int i = 0;
		float coef = 1.0f;
		float delta;

		while (i < fidx) {
			coef *= (1.02f);
			i++;
		}

		delta = pow(coef, 1.0f / 8.0f);
		K[15] = coef;

		for (i = 1; i < 16; i++) {
			K[15 - i] = K[15 - i + 1] / delta;
		}
	}

void strikeIntervalInfer(float* K, float T) {

	float fidx = T * 12.0f + 1.0f;
	int i = 0;
	float coef = 1.0f;
	float delta;

	while (i < fidx) {
		coef *= (1.02f);
		i++;
	}

	delta = pow(coef, 1.0f / 3.0f);
	K[5] = coef;

	for (i = 1; i < 6; i++) {
		K[5 - i] = K[5 - i + 1] / delta;
	}
}

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

	float Y0[6] = {logf(0.35f), logf(0.27f), logf(0.2f), logf(0.16f), logf(0.12f), logf(0.08f)};
	float m[6] = {logf(0.3f), logf(0.24f), logf(0.18f), logf(0.13f), logf(0.09f), logf(0.06f)};
	float alpha[8] = {0.2f, 0.4f, 0.8f, 1.6f, 3.2f, 6.4f, 12.8f, 25.6f};
		float nu2[6] = {0.5f, 0.7f, 0.9f, 1.1f, 1.3f, 1.5f};
	float rho[6] = {0.05f, -0.15f, -0.35f, -0.55f, -0.75f, -0.95f};
	
	float Tmt[16] = { 1.0f / 12.0f,  2.0f / 12.0f, 3.0f / 12.0f, 4.0f / 12.0f, 5.0f / 12.0f, 6.0f / 12.0f, 7.0f / 12.0f,
					  8.0f / 12.0f, 9.0f / 12.0f, 10.0f / 12.0f, 11.0f / 12.0f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f };

	//float Tmt[5] = { 3.0f / 12.0f, 6.0f / 12.0f, 1.0f, 1.5f, 2.0f };


	float Str[16];
	//float Str[6];

	cudaMemcpyToSymbol(Y0d, Y0, 6*sizeof(float));
	cudaMemcpyToSymbol(md, m, 6*sizeof(float));
	cudaMemcpyToSymbol(alphad, alpha, 8*sizeof(float));
	cudaMemcpyToSymbol(nu2d, nu2, 6*sizeof(float));
	cudaMemcpyToSymbol(rhod, rho, 6*sizeof(float));

	int pidx, same;
	int NTPB = 256 * 2; //256; // 256 * 2;
	int NB =  81 * 4; // 81*3; // 81 * 4;
	int Ntraj = 32 * 512; // 256 * 512; // 32 * 512;
	float dt = sqrtf(1.0f/(1000.0f));
	float StrR, mR, alphaR, betaR, rhoR, price, error;

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
	for(int i=0; i<16; i++){
//	for (int i = 0; i < 5; i++) {
		strikeInterval(Str, Tmt[i]);
		//strikeIntervalInfer(Str, Tmt[i]);
		cudaMemcpyToSymbol(Strd, Str, 16*sizeof(float));
		//cudaMemcpyToSymbol(Strd, Str, 6 * sizeof(float));
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