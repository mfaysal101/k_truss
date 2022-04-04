###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/home/packages/cuda/11.2.2/l2rjomqd

###########################################################

## CXX COMPILER OPTIONS ##

# CXX compiler options:
CXX = g++
LINK = $(CXX)
LFLAGS =  -g -fopenmp -Wall -Werror -O3	-std=c++11

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -arch=compute_35 -code=sm_35
# NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include

##########################################################

## Make variables ##

# Target executable name:
EXE = ktruss

# Object files:
OBJS = $(OBJ_DIR)/Ktruss.o $(OBJ_DIR)/MyGraph.o $(OBJ_DIR)/TecIndexSB.o $(OBJ_DIR)/cudakernel.o


##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(LINK) $(LFLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile Ktruss.cpp file to object files:
$(OBJ_DIR)/%.o : %.cpp
	$(LINK) $(LFLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.h
	$(LINK) $(LFLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

all: $(EXE)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(EXE)




