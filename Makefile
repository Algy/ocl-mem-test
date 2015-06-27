CC = clang 

memtest: host.c sync.c tsoff.c
	$(CC) -std=c99 -pthread -g -O3 -Wall host.c sync.c tsoff.c -lm -lOpenCL -o memtest

memtest_nobarrier: host.c sync.c
	$(CC) -std=c99 -D NO_BARRIER -pthread -g -Wall host.c sync.c -lOpenCL -o memtest


run_gpu0: memtest
	sudo sh -c "export COMPUTE_PROFILE=1;export COMPUTE_PROFILE_CONFIG=/home/alchan/.nvidia-cmd-prof-config; ./memtest --gpu-flag 3 --mb 700; chown alchan:alchan opencl_profile_*"

run_gpu1: memtest
	sudo sh -c "export COMPUTE_PROFILE=1;export COMPUTE_PROFILE_CONFIG=/home/alchan/.nvidia-cmd-prof-config; ./memtest 2 700; chown alchan:alchan opencl_profile_*"
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot

run_multi: memtest
	sudo sh -c "export COMPUTE_PROFILE=1;export COMPUTE_PROFILE_CONFIG=/home/alchan/.nvidia-cmd-prof-config;./memtest 3 700; chown alchan:alchan opencl_profile_*"
	./plotter.py
	cat cldat.dat
	gnuplot clprof.plot
clean:
	rm -f memtest
	rm -f offtest
