# cuda-variance-gamma-process
Course project to code in cuda a parallelized Monte Carlo generator for estimating the price of a variance gamma process call option.

The MC.cu file generates prices estimation using nested Monte Carlo :

<pre>nvcc MC.cu -o MC
./MC</pre>

The notebook [train.ipynb](train.ipynb) trains a neural net that approximates the option prices.

The [presentation slides](CUDA_variance_gamma.pdf) for the project.

