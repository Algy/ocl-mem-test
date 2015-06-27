CC = clang 

memtest: host.c sync.c tsoff.c
	$(CC) -std=c99 -pthread -g -O3 -Wall host.c sync.c tsoff.c -lm -lrt -lOpenCL -o memtest

clean:
	rm -f memtest
	rm -f offtest
