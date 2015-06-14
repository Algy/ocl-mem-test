CC = clang all: memtest

memtest: host.c sync.c
	$(CC) -std=c99 -pthread -g -Wall host.c sync.c -lOpenCL -o memtest

offtest: tsoff.c
	$(CC) -std=c99 -g -Wall tsoff.c -lOpenCL -o offtest

run_gpu0: memtest offtest
	sudo sh -c "export COMPUTE_PROFILE=1;export COMPUTE_PROFILE_CONFIG=/home/alchan/.nvidia-cmd-prof-config; ./memtest 1 0"
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot

run_gpu1: memtest offtest
	sudo sh -c "export COMPUTE_PROFILE=1;export COMPUTE_PROFILE_CONFIG=/home/alchan/.nvidia-cmd-prof-config; ./memtest 2 0"
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot

run_multi: memtest offtest
	sudo sh -c "export COMPUTE_PROFILE=1;export COMPUTE_PROFILE_CONFIG=/home/alchan/.nvidia-cmd-prof-config;./memtest 3 10"
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot
clean:
	rm -f memtest
	rm -f offtest
