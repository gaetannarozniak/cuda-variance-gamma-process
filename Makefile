# Simple Makefile for the Johnk’s Gamma generator in CUDA

NVCC       = nvcc
TARGET     = johnk_gamma
SRC        = johnk_gamma.cu
OBJ        = johnk_gamma.o
CUDARFLAGS = -arch=sm_52  # or whichever your GPU supports

$(TARGET): $(OBJ)
	$(NVCC) $(CUDARFLAGS) -o $(TARGET) $(OBJ) -lcurand

$(OBJ): $(SRC)
	$(NVCC) $(CUDARFLAGS) -c $(SRC) -o $(OBJ) -lcurand

clean:
	rm -f $(OBJ) $(TARGET)
