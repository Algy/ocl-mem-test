CC = clang
all: memtest

memtest: host.c sync.c
	$(CC) -std=c99 -pthread -g -Wall host.c sync.c -lOpenCL -o memtest

offtest: tsoff.c
	$(CC) -std=c99 -g -Wall tsoff.c -lOpenCL -o offtest

run_gpu0: memtest
	sudo ./memtest 1 0
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot

run_gpu1: memtest
	sudo ./memtest 2 0
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot

run_multi: memtest
	sudo ./memtest 3 0
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot
clean:
	rm -f memtest
	rm -f offtest
