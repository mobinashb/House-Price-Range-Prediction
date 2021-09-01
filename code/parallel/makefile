CC=g++
CF=-std=c++11

all: HousePricePrediction.out
		make HousePricePrediction.out
		make clean

HousePricePrediction.out: main.o dataProcessor.o
	$(CC) $(CF) main.o dataProcessor.o -o HousePricePrediction.out

main.o: main.cpp
	$(CC) $(CF) -c main.cpp

dataProcessor.o: dataProcessor.cpp
	$(CC) $(CF) -c dataProcessor.cpp

clean:
	rm *.o
