# Various flags
CXX  = CC
LINK = $(CXX)

CXXFLAGS = -g -Wall -O3 -fopenmp --std=c++11  #-I #-Wall -O3 -funroll-loops -pipe  
LFLAGS =  -g -fopenmp -Wall -Werror -O3


TARGET  = ktruss

HEADER  = Edge.h MyGraph.h SGN.h TecIndexSB.h
FILES = Ktruss.cpp MyGraph.cpp TecIndexSB.cpp

OBJECTS = $(FILES:.cpp=.o)

$(TARGET): ${OBJECTS}
	$(LINK) $(LFLAGS) $^ -o $@ 

all: $(TARGET)

clean:
	rm -f $(OBJECTS)

distclean:
	rm -f $(OBJECTS) $(TARGET)

# Compile and dependency
$(OBJECTS): $(HEADER) Makefile








