# Makefile
DEBUG = 0


LU_INC = -Ipreprocess -I../include -Inicslu/include

PREPROCESS_LIBS = preprocess/preprocess.o nicslu/lib/nicslu.a nicslu/util/nicslu_util.a

NVCC = nvcc
CUDA_ARCH ?= sm_60
CPP = g++
EXE = lu_cmd

CFLAGS = -O3 -m64 -Wall -std=c++17 $(LU_INC) -DGLU_DEBUG=${DEBUG} -MMD -MP
NVCCFLAGS = -O3 -std=c++17 -arch=$(CUDA_ARCH) -Xcompiler $(LU_INC) -DGLU_DEBUG=${DEBUG} -MMD -MP

MAIN: $(EXE)

-include $(wildcard *.d)

lu_cmd: Timer.o numeric.o symbolic.o lu_cmd.o $(PREPROCESS_LIBS)
	$(NVCC) $^ -o $@

Timer.o: Timer.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

numeric.o: numeric.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

symbolic.o: symbolic.cc
	$(CPP) $(CFLAGS) -c $< -o $@

lu_cmd.o: lu_cmd.cpp
	$(CPP) $(CFLAGS) -c $< -o $@

preprocess/preprocess.o:
	make -C preprocess/

nicslu/lib/nicslu.a:
	make -C nicslu/

nicslu/lib/nicslu_util.a:
	make -C nicslu/

clean:
	rm -f *.o *.d *~ *.so $(EXE)
	make -C preprocess/ clean

distclean:
	rm -f *.o *.d *~ *.so $(EXE)
	make -C nicslu/ clean
	make -C preprocess/ clean


