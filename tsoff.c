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


int main() {
    srand(time(NULL));
    /*
    printf("Setting scheduler policy to FIFO 99...\n");
    struct sched_param sched_param;
    sched_param.sched_priority = 99;
    if (sched_setscheduler(0, SCHED_FIFO, &sched_param)) {
        printf("Failed to set scheduler policy. Are you root?\n");
        return 1;
    }
    */
    
    int idx;
    cl_int err;
    cl_platform_id platform_id;
    cl_uint nplatforms, ndevices;

    err = clGetPlatformIDs(1, &platform_id, &nplatforms);
    CHECK_ERROR;

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &ndevices);
    CHECK_ERROR;

    cl_device_id device_ids[ndevices];
    cl_command_queue queues[ndevices];

    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, ndevices, device_ids, NULL);
    CHECK_ERROR;

    cl_context context = clCreateContext(NULL, ndevices, device_ids, NULL, NULL, &err);
    CHECK_ERROR;

    for (idx = 0; idx < ndevices; idx++) {
        queues[idx] = clCreateCommandQueue(context, device_ids[idx], CL_QUEUE_PROFILING_ENABLE, &err);
        CHECK_ERROR;
    }

    cl_mem dummy_mems[ndevices];

    for (idx = 0; idx < ndevices; idx++) {
        dummy_mems[idx] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * 1, NULL, &err);
        CHECK_ERROR;
    }

    cl_uchar c[1] = {'A', };
    for (idx = 0; idx < ndevices; idx++) {
        err = clEnqueueWriteBuffer(queues[idx], dummy_mems[idx], CL_FALSE, 0, sizeof(cl_uchar) * 1, c, 0, NULL, NULL);
        CHECK_ERROR;
    }

    for (idx = 0; idx < ndevices; idx++) {
        clFlush(queues[idx]);
    }

    for (idx = 0; idx < ndevices; idx++) {
        clFinish(queues[idx]);
    }

    return 0;
}