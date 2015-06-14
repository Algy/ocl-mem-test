#define _GNU_SOURCE
#include <stdio.h> 
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include <pthread.h>
#include <sched.h>
#include <assert.h>

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

#include "sync.h"

const char *getErrorString(cl_int error) {
    switch(error){
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}
#define CHECK_ERROR do { \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "[OpenCL]%s at [%s:%d]\n", getErrorString(err), __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)



const char *inc_source =
    "__kernel void inc(__global uchar *arr, uint size) {\n" \
    "   int idx = get_global_id(0); \n" \
    "   int idy;\n" \
    "   if (idx < size) \n" \
    "       for (idy = 0; idy < 1000; idy++) \n" \
    "           arr[idx] *= 5; \n" \
    "}";

typedef struct {
    cl_platform_id platform_id;
    cl_context contexts[2];
    cl_command_queue queues[2];
    cl_program programs[2];
    cl_kernel knl_inc[2];
} Memtest;

static void Memtest_init(Memtest *memtest) {
    int idx;
    cl_int err;
    cl_uint platform_num, ndevices;
    cl_device_id device_ids[2];

    err = clGetPlatformIDs(1, &memtest->platform_id, &platform_num);
    CHECK_ERROR;

    err = clGetDeviceIDs(memtest->platform_id, CL_DEVICE_TYPE_GPU, 2, device_ids, &ndevices);
    CHECK_ERROR;


    for (idx = 0; idx < ndevices; idx++) {
        char device_name[256];
        clGetDeviceInfo(device_ids[idx], CL_DEVICE_NAME, 256, device_name, NULL);
        fprintf(stderr, "device [%d]: %s\n", idx, device_name);
    }

    assert(ndevices >= 2);
    for (idx = 0; idx < 2; idx++) {
        memtest->contexts[idx] = clCreateContext(NULL, 2, device_ids, NULL, NULL, &err);
        CHECK_ERROR;
    }
    printf("# of devices: %d\n", ndevices);

    for (idx = 0; idx < 2; idx++) {
        memtest->queues[idx] = clCreateCommandQueue(memtest->contexts[idx], device_ids[idx], CL_QUEUE_PROFILING_ENABLE, &err);
        CHECK_ERROR;
        memtest->programs[idx] = clCreateProgramWithSource(memtest->contexts[idx], 1, (const char **)&inc_source, NULL, &err);
        CHECK_ERROR;
    }

    for (idx = 0; idx < 2; idx++) {
        err = clBuildProgram(memtest->programs[idx], 0, NULL, NULL, NULL, NULL);
        CHECK_ERROR;
        memtest->knl_inc[idx] = clCreateKernel(memtest->programs[idx], "inc", &err);
        CHECK_ERROR;
    }
}

void Memtest_remove(Memtest* memtest) {
    int idx;
    for (idx = 0; idx < 2; idx++) {
        clReleaseKernel(memtest->knl_inc[idx]);
        clReleaseProgram(memtest->programs[idx]);
    }
    for (idx = 0; idx < 2; idx++) {
        clReleaseCommandQueue(memtest->queues[idx]);
    }
    for (idx = 0; idx < 2; idx++) {
        clReleaseContext(memtest->contexts[idx]);
    }
}

typedef struct {
    Memtest *memtest;
    int id;
    size_t memsize;
    SpinningBarrier *barrier;
    cpu_set_t affinity_mask;
} MemtestContext;

static void unit_work(MemtestContext *ctx) {
#ifndef NO_BARRIER
# define setup_barrier sb_spin(ctx->barrier)
#else 
# define setup_barrier 
#endif
    cl_int err;

    // BARRIER
    setup_barrier;

    size_t memsize = ctx->memsize;
    cl_context cl_ctx = ctx->memtest->contexts[ctx->id];
    cl_command_queue queue = ctx->memtest->queues[ctx->id];
    cl_kernel kernel = ctx->memtest->knl_inc[ctx->id];

    cl_uchar *buffer = calloc(sizeof(cl_uchar), ctx->memsize);
    if (!buffer) {
        printf("Out of Memory\n");
        abort();
    }

    // BARRIER
    setup_barrier;

    cl_mem mem_arr;
    mem_arr = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uchar) * memsize, NULL, &err);
    CHECK_ERROR;
    printf("Created a buffer\n");
    printf("[%d]Starting to transfer the buffer to each device..\n", ctx->id);

    // BARRIER
    setup_barrier;

    clock_t st_t = clock();

    err = clEnqueueWriteBuffer(queue, mem_arr, CL_TRUE, 0, sizeof(cl_uchar) * memsize, buffer, 0, NULL, NULL); 
    CHECK_ERROR;

    // BARRIER
    setup_barrier;

    printf("Transfered the buffer to each device..(%.3lf)\n", ((double)st_t / CLOCKS_PER_SEC) * 1000);

    size_t gblsize[1] = {memsize};
    size_t lclsize[1] = {128};

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_arr);
    CHECK_ERROR;
    cl_uint _size = memsize;
    err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &_size);
    CHECK_ERROR;

    // BARRIER
    setup_barrier;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gblsize, lclsize, 0, NULL, NULL);
    CHECK_ERROR;
    clFlush(queue);

    // BARRIER
    setup_barrier;

    err = clEnqueueReadBuffer(queue, mem_arr, CL_TRUE, 0, sizeof(cl_uchar) * memsize, buffer, 0, NULL, NULL); 
    CHECK_ERROR;

    free(buffer);
    clReleaseMemObject(mem_arr);
}

static void* kernel_work(void *arg) {
    int idx;
    MemtestContext *ctx = (MemtestContext *)arg;
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &ctx->affinity_mask);
    for (idx = 0; idx < 3; idx++)
        unit_work(ctx);
    return NULL;
}

static void Memtest_run(Memtest *memtest, unsigned int gpu_flag, size_t memsize) {
    int idx;
    unsigned int nthread = 0;
    int gpu_max = 0;

    unsigned int gpu_iter;
    for (gpu_iter = gpu_flag; gpu_iter; gpu_iter >>= 1) {
        gpu_max++;
        if (gpu_iter & 1)
            nthread++;
    }

    SpinningBarrier shared_barrier;

    sb_init(&shared_barrier, nthread);

    MemtestContext contexts[nthread];

    int cnt = 0;
    for (idx = 0; idx < gpu_max; idx++) {
        if (gpu_flag & (1u << idx)) {
            contexts[cnt].memtest = memtest;
            contexts[cnt].id = idx;
            contexts[cnt].memsize = memsize;
            contexts[cnt].barrier = &shared_barrier;
            CPU_ZERO(&contexts[cnt].affinity_mask);
            CPU_SET(cnt, &contexts[cnt].affinity_mask);
            cnt++;
        }
    }

    int nsubthread = nthread - 1;
    pthread_t subthreads[nsubthread];
    for (idx = 0; idx < nsubthread; idx++) {
        pthread_create(&subthreads[idx], NULL, kernel_work, &contexts[idx + 1]);
    }
    kernel_work(&contexts[0]);

    for (idx = 0; idx < nsubthread; idx++) {
        pthread_join(subthreads[idx], NULL);
    }
}

int main(int argc, char** argv) {
    srand(time(NULL));
    if (argc < 3) {
        printf("Usage -- [program] flag memsize\n");
        return 0;
    }
    printf("Setting scheduler policy to FIFO 99...\n");
    struct sched_param sched_param;
    sched_param.sched_priority = 99;
    if (sched_setscheduler(0, SCHED_FIFO, &sched_param)) {
        printf("Failed to set scheduler policy. Are you root?\n");
        return 1;
    }

    unsigned int gpu_flag = (unsigned int)strtol(argv[1], NULL, 10);
    size_t memsize = (size_t)strtol(argv[2], NULL, 10) * 1024 * 1024;

    int gpu_idx = 0;
    unsigned int gpu_iter;
    for (gpu_iter = gpu_flag; gpu_iter; gpu_iter >>= 1) {
        if (gpu_iter & 1)
            printf("Use GPU%d\n", gpu_idx);
        gpu_idx++;
    }
    printf("Memory size: %zu MB\n", memsize / 1024 / 1024);
    if (memsize == 0) {
        memsize = 512;
    }
    
    Memtest memtest;
    Memtest_init(&memtest);
    Memtest_run(&memtest, gpu_flag, memsize);
    Memtest_remove(&memtest);
    return 0;
}
