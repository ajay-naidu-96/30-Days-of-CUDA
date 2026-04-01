TARGET = axpy
SRC = cuBlas_vec_add.cu

NVCC ?= nvcc
CCBIN = $(CONDA_PREFIX)/bin/x86_64-conda-linux-gnu-g++

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -ccbin $(CCBIN) -O2 $(SRC) -o $(TARGET) -lcublas -lstdc++

clean:
	rm -f $(TARGET)