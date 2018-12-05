CUDA = /usr/local/cuda
NVCC = $(CUDA)/bin/nvcc -O3
LINK = $(addprefix -Xlinker , -lm -rpath $(CUDA)/lib)
FLGS = $(addprefix --compiler-options , -Wall)

compile: cpu gpu

cpu:
	gcc main.c 1body.c -lm -O0 -Wall -o 1body0
	gcc main.c 1body.c -lm -O3 -Wall -o 1body3
	gcc main.c 2body.c -lm -O0 -Wall -o 2body0
	gcc main.c 2body.c -lm -O3 -Wall -o 2body3
	gcc main.c 3body.c -lm -O0 -Wall -o 3body0
	gcc main.c 3body.c -lm -O3 -Wall -o 3body3
	gcc main.c nbody.c -lm -O0 -Wall -o nbody0
	gcc main.c nbody.c -lm -O3 -Wall -o nbody3

gpu:
	$(NVCC) main.c io.cu kern.cu $(LINK) $(FLGS) -o nbodyg

clean:
	-rm -f ?body?
	-rm -f *.out
