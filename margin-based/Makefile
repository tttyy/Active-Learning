CXX ?= g++ -arch x86_64
CC ?= gcc -arch x86_64
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a
SHVER = 1
#LIBS = -lblas

all: margin

margin: margin-main.o margin.o datapoint.o tron.o linear.o blas/blas.a
	$(CXX) $(CFLAGS) -o margin margin-main.o margin.o datapoint.o tron.o linear.o blas/blas.a

margin-main.o: margin-main.cpp
	$(CXX) $(CFLAGS) -c -o margin-main.o margin-main.cpp

margin.o: MarginActiveLearning.cpp MarginActiveLearning.h
	$(CXX) $(CFLAGS) -c -o margin.o MarginActiveLearning.cpp

datapoint.o: datapoint.cpp datapoint.h
	$(CXX) $(CFLAGS) -c -o datapoint.o datapoint.cpp

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a:
	cd blas; make OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	cd blas;	make clean
	rm -f *~ tron.o linear.o wrapper.o liblinear.so.$(SHVER) margin datapoint.o
