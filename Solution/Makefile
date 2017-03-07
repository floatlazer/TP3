CXX = g++ -std=c++11
LIBS = -lm 
CFLAGS = -Ilodepng -O3 -march=native  -fopenmp

OBJS = Matrix.o TestProduct.o ProdMatMat.o
OBJS21= dotproduct_omp.o
OBJS22= dotproduct_threads.o
OBJS3= Vecteur.cpp bitonic.cpp
OBJS4= bhudda.cpp lodepng/lodepng.cpp

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $<

all:	clean TestProduct.exe dotproduct.exe dotproduct_omp.exe dotproduct_threads.exe bitonic.exe bhudda.exe

clean:
	@rm -fr *.o *.exe *~

TestProduct.exe: $(OBJS)
	$(CXX) $(CFLAGS) -fopenmp -o $@ $(OBJS) $(LIBS)

dotproduct_omp.exe: $(OBJS21)
	$(CXX) $(CFLAGS) -fopenmp -o $@ $(OBJS21) $(LIBS)

dotproduct.exe: $(OBJS21)
	$(CXX) $(CFLAGS) -o $@ $(OBJS21) $(LIBS)

dotproduct_threads.exe: $(OBJS22)
	$(CXX) $(CFLAGS) -o $@ $(OBJS22) $(LIBS) -lpthread

bitonic.exe:	$(OBJS3)
	$(CXX) $(CFLAGS) -o $@ $(OBJS3) $(LIBS)

bhudda.exe:	$(OBJS4)
	$(CXX) $(CFLAGS) -o $@ $(OBJS4) $(LIBS)
