GCC = g++
INCLUDE = $(HOME)/studio/usr/include
APP = tensor
FLAGS = -std=c++14 -Wall

$(APP) : main.o
	$(GCC) -o $(APP) main.o

main.o : main.cpp u-tensor.hpp u-shape.hpp u-dtype.hpp u-op.hpp op/u-op-cpu.hpp u-mm.hpp mm/u-mm-cpu.hpp
	$(GCC) -c -o main.o main.cpp -I $(INCLUDE) $(FLAGS)

#shape.o : u-shape.hpp
#    $(GCC) -c -o shape.o u-shape.hpp

#dtype.o : u-dtype.hpp
#    $(GCC) -c -o dtype.o u-dtype.hpp

#op.o : u-op.hpp mm/u-op-cpu.hpp
#    $(GCC) -c -o op.o u-op.hpp mm/u-op-cpu.hpp

clean :
	rm *.o tensor
