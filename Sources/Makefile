CXX = g++ -std=c++11
LIBS = -lm -lpthread
CFLAGS = -Ilodepng  -fopenmp  -O1 -march=native

OBJS = Matrix.o TestProduct.o ProdMatMat.o
OBJS2= dotproduct.o
OBJS3= Vecteur.cpp bitonic.cpp
OBJS4= bhudda.cpp lodepng/lodepng.cpp

%.o: %.cpp
	$(CXX) $(CFLAGS) -c $<

all:	clean TestProduct.exe dotproduct.exe bitonic.exe bhudda.exe

clean:
	@rm -fr *.o *.exe *~

TestProduct.exe: $(OBJS)
	$(CXX) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

dotproduct.exe: $(OBJS2)
	$(CXX) $(CFLAGS) -o $@ $(OBJS2) $(LIBS)

bitonic.exe:	$(OBJS3)
	$(CXX) $(CFLAGS) -o $@ $(OBJS3) $(LIBS)

bhudda.exe:	$(OBJS4)
	$(CXX) $(CFLAGS) -o $@ $(OBJS4) $(LIBS)
