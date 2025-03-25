test: tokenizer.o main.o
	g++ tokenizer.o main.o -o test
tokenizer.o: tokenizer.cpp
	g++ -c tokenizer.cpp
main.o: main.cpp
	g++ -c main.cpp
clean:
	rm -f *.o test