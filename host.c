#define _GNU_SOURCE
#define _XOPEN_SOURCE
#include <stdio.h> 
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <getopt.h>
#include <limits.h>
#include <errno.h>

#include <pthread.h>
#include <sched.h>
#include <assert.h>

#ifdef __APPLE__
# include <OpenCL/cl.h>
#else
# include <CL/cl.h>
#endif

#include "sync.h"
#include "tsoff.h"

#define MAX_GPU 32

static const char *getErrorString(cl_int error) {
    switch(error) {
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

static ssize_t parse_colon_str(char *s, size_t max_cnt, long long *out);
struct {
    bool fill_zero;

    size_t memsize[MAX_GPU];

    bool use_ts_profile_file;
    char ts_profile_path[1024];

    bool use_ts_profile_str;
    char ts_profile_str[1024]; /* sperated by ":", like 202:-22902:22... (in nanoseconds) */

    bool use_rw_offsets;
    char rw_offsets_str[1024]; /* same as above */

    unsigned long long gpu_flag;

    bool use_fifo99;

    int run_count;

    bool mixed_mode;
    int leader_id;
} program_option;

typedef struct {
    int gpu_id;
    char *op_name;
    long long queued_time;
    long long submitted_time;
    long long start_time;
    long long end_time;
    long long exec_time;
} GpuLogRecord;

#define MAX_LOG 1024
typedef struct {
    GpuLogRecord records_per_gpu[MAX_GPU][MAX_LOG];
    size_t gpu_log_cnt[MAX_GPU];
    bool is_flooded[MAX_GPU];
    size_t gpu_cnt;
} GpuLog;

static void GpuLog_init(GpuLog *logger, size_t gpu_cnt) {
    int idx;

    logger->gpu_cnt = gpu_cnt;
    for (idx = 0; idx < gpu_cnt; idx++) {
        logger->is_flooded[idx] = false;
        logger->gpu_log_cnt[idx] = 0;
    }
}

static void GpuLog_log(GpuLog *logger, int gpu_id, char *op_name, long long ts_delta, cl_event ev) {
    cl_int err;

    size_t log_cnt = logger->gpu_log_cnt[gpu_id];
    if (log_cnt >= MAX_LOG)
        logger->is_flooded[gpu_id] = true;
    else {
        cl_ulong queued_time, submitted_time, start_time, end_time;
        err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, NULL);
        CHECK_ERROR;
        err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitted_time, NULL);
        CHECK_ERROR;
        err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
        CHECK_ERROR;
        err = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
        CHECK_ERROR;

        if (queued_time == 0) {
            fprintf(stderr, "[Warning] GPU%d %s: queued_time is 0\n", gpu_id, op_name);
        }
        if (submitted_time == 0) {
            fprintf(stderr, "[Warning] GPU%d %s: submitted_time is 0\n", gpu_id, op_name);
        }
        if (start_time == 0) {
            fprintf(stderr, "[Warning] GPU%d %s: start_time is 0\n", gpu_id, op_name);
        }
        if (end_time == 0) {
            fprintf(stderr, "[Warning] GPU%d %s: end_time is 0\n", gpu_id, op_name);
        }

        logger->records_per_gpu[gpu_id][log_cnt] = (GpuLogRecord) {
            .gpu_id = gpu_id,
            .op_name = op_name,
            .queued_time = (long long)queued_time - ts_delta,
            .submitted_time = (long long)submitted_time - ts_delta,
            .start_time = (long long)start_time - ts_delta,
            .end_time = (long long)end_time - ts_delta,
            .exec_time = (long long)(end_time - start_time)
        };
        logger->gpu_log_cnt[gpu_id]++;
    }
}

typedef struct {
    cl_platform_id platform_id;
    size_t ndevices;

    cl_context contexts[MAX_GPU];
    cl_command_queue queues[MAX_GPU];
    cl_program programs[MAX_GPU];
    cl_kernel knl_inc[MAX_GPU];

} Memtest;

static void Memtest_init(Memtest *memtest) {
    int idx;
    cl_int err;
    cl_uint platform_num, ndevices;

    err = clGetPlatformIDs(1, &memtest->platform_id, &platform_num);
    CHECK_ERROR;

    err = clGetDeviceIDs(memtest->platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &ndevices);
    CHECK_ERROR;

    if (ndevices >= MAX_GPU)
        ndevices = MAX_GPU;
    fprintf(stderr, "# of devices: %d\n", ndevices);
    memtest->ndevices = ndevices;

    cl_device_id device_ids[ndevices];

    err = clGetDeviceIDs(memtest->platform_id, CL_DEVICE_TYPE_GPU, ndevices, device_ids, NULL);
    CHECK_ERROR;


    for (idx = 0; idx < ndevices; idx++) {
        char device_name[256];
        clGetDeviceInfo(device_ids[idx], CL_DEVICE_NAME, 256, device_name, NULL);
        fprintf(stderr, "device [%d]: %s\n", idx, device_name);
    }

    for (idx = 0; idx < ndevices; idx++) {
        memtest->contexts[idx] = clCreateContext(NULL, ndevices, device_ids, NULL, NULL, &err);
        CHECK_ERROR;
    }

    for (idx = 0; idx < ndevices; idx++) {
        memtest->queues[idx] = clCreateCommandQueue(memtest->contexts[idx], device_ids[idx], CL_QUEUE_PROFILING_ENABLE, &err);
        CHECK_ERROR;
        memtest->programs[idx] = clCreateProgramWithSource(memtest->contexts[idx], 1, (const char **)&inc_source, NULL, &err);
        CHECK_ERROR;
    }

    for (idx = 0; idx < ndevices; idx++) {
        err = clBuildProgram(memtest->programs[idx], 0, NULL, NULL, NULL, NULL);
        CHECK_ERROR;
        memtest->knl_inc[idx] = clCreateKernel(memtest->programs[idx], "inc", &err);
        CHECK_ERROR;
    }
}

void Memtest_remove(Memtest* memtest) {
    int idx;
    size_t ndevices = memtest->ndevices;
    for (idx = 0; idx < ndevices; idx++) {
        clReleaseKernel(memtest->knl_inc[idx]);
        clReleaseProgram(memtest->programs[idx]);
    }
    for (idx = 0; idx < ndevices; idx++) {
        clReleaseCommandQueue(memtest->queues[idx]);
    }
    for (idx = 0; idx < ndevices; idx++) {
        clReleaseContext(memtest->contexts[idx]);
    }
}

static void spin_wait(long long rw_offset) {
    if (rw_offset > 0) {
        struct timespec st_ts;
        clock_gettime(CLOCK_MONOTONIC, &st_ts);
        while (1) {
            struct timespec cur_ts;
            if (clock_gettime(CLOCK_MONOTONIC, &cur_ts) == -1) {
                perror("clock_gettime()");
                abort();
            }
            long long elapsed_ms = (cur_ts.tv_sec - st_ts.tv_sec) * 1000LL + (cur_ts.tv_nsec - st_ts.tv_nsec) / 1000000LL;
            int idx;
            if (elapsed_ms >= rw_offset)
                break;
            for (idx = 0; idx < 10000; idx++); // spin for a while
        }
    }
}

typedef struct {
    bool randomize_buffer;
    GpuLog *logger;
    size_t memsize[MAX_GPU];

    long long gpu_ts_profile[MAX_GPU];
    long long rw_offsets[MAX_GPU];

    unsigned long long gpu_flag;
    int run_count;

    bool mixed_mode;
    int leader_id;
} MemtestSharedInfo;

typedef struct {
    Memtest *memtest;
    MemtestSharedInfo *shared_info;

    int id;
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
    int idx;

    bool mixed_mode = ctx->shared_info->mixed_mode;
    int leader_id = ctx->shared_info->leader_id;

    // BARRIER
    setup_barrier;

    int gpu_id = ctx->id;
    size_t memsize = ctx->shared_info->memsize[gpu_id];
    cl_context cl_ctx = ctx->memtest->contexts[gpu_id];
    cl_command_queue queue = ctx->memtest->queues[gpu_id];
    cl_kernel kernel = ctx->memtest->knl_inc[gpu_id];

    long long ts_delta = ctx->shared_info->gpu_ts_profile[gpu_id];
    long long rw_offset_min = LLONG_MAX;
    for (idx = 0; idx < ctx->memtest->ndevices; idx++) {
        if (rw_offset_min > ctx->shared_info->rw_offsets[idx])
            rw_offset_min = ctx->shared_info->rw_offsets[idx];
    }
    long long rw_offset = ctx->shared_info->rw_offsets[gpu_id] - rw_offset_min;


    cl_uchar *buffer = calloc(memsize, sizeof(cl_uchar));
    if (!buffer) {
        fprintf(stderr, "Out of Memory\n");
        abort();
    }

    if (ctx->shared_info->randomize_buffer) {
        unsigned int seed = (unsigned int)time(NULL);
        cl_uchar* b_iter, *b_end;
        b_end = buffer + memsize;
        for (b_iter = buffer; b_iter < b_end; b_iter++) {
            *b_iter = (cl_uchar)rand_r(&seed);
        }
    }

    // BARRIER
    setup_barrier;

    cl_mem mem_arr;
    mem_arr = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(cl_uchar) * memsize, NULL, &err);
    CHECK_ERROR;
    fprintf(stderr, "Created a buffer\n");
    fprintf(stderr, "[%d]Starting to transfer the buffer to each device..\n", gpu_id);

    // BARRIER
    setup_barrier;
    if (mixed_mode && gpu_id != leader_id) {
        setup_barrier;
        setup_barrier;
        setup_barrier;
    }
    spin_wait(rw_offset);
    clock_t st_t = clock();

    cl_event h2d_ev, d2h_ev, kernel_ev;
    err = clEnqueueWriteBuffer(queue, mem_arr, CL_TRUE, 0, sizeof(cl_uchar) * memsize, buffer, 0, NULL, &h2d_ev);
    CHECK_ERROR;

    clFinish(queue);
    clWaitForEvents(1, &h2d_ev);
    GpuLog_log(ctx->shared_info->logger, gpu_id, "H2DMemcpy", ts_delta, h2d_ev);
    clReleaseEvent(h2d_ev);

    // BARRIER
    setup_barrier;

    fprintf(stderr, "Transfered the buffer to each device..(%.3lf)\n", ((double)st_t / CLOCKS_PER_SEC) * 1000);

    size_t gblsize[1] = {memsize};
    size_t lclsize[1] = {memsize > 128? 128 : memsize};

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mem_arr);
    CHECK_ERROR;
    cl_uint _size = memsize;
    err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &_size);
    CHECK_ERROR;

    // BARRIER
    setup_barrier;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gblsize, lclsize, 0, NULL, &kernel_ev);
    CHECK_ERROR;

    clFinish(queue);
    clWaitForEvents(1, &kernel_ev);
    GpuLog_log(ctx->shared_info->logger, gpu_id, "kernel_inc", ts_delta, kernel_ev);
    clReleaseEvent(kernel_ev);

    // BARRIER
    setup_barrier;
    spin_wait(rw_offset);

    err = clEnqueueReadBuffer(queue, mem_arr, CL_TRUE, 0, sizeof(cl_uchar) * memsize, buffer, 0, NULL, &d2h_ev); 
    CHECK_ERROR;

    clFinish(queue);
    clWaitForEvents(1, &d2h_ev);
    GpuLog_log(ctx->shared_info->logger, gpu_id, "D2HMemcpy", ts_delta, d2h_ev);
    clReleaseEvent(d2h_ev);

    free(buffer);
    clReleaseMemObject(mem_arr);
    if (mixed_mode && gpu_id == leader_id) {
        setup_barrier;
        setup_barrier;
        setup_barrier;
    }
#undef setup_barrier
}

static void* kernel_work(void *arg) {
    int idx;
    MemtestContext *ctx = (MemtestContext *)arg;
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &ctx->affinity_mask)) {
        fprintf(stderr, "Warning: Failed to set affinity of thread assigned to gpu %d[%s:%d]\n", ctx->id, __FILE__, __LINE__);
    }
    for (idx = 0; idx < ctx->shared_info->run_count; idx++)
        unit_work(ctx);
    return NULL;
}

static void Memtest_run(Memtest *memtest, MemtestSharedInfo *shared_info) {
    int idx;
    unsigned int nthread = 0;
    int gpu_max = 0;

    unsigned long long gpu_iter, gpu_flag;

    gpu_flag = shared_info->gpu_flag;
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
            contexts[cnt] = (MemtestContext) {
                .memtest = memtest,
                .shared_info = shared_info,
                .id = idx,
                .barrier = &shared_barrier
            };
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


static void print_option() {
    fprintf(stderr, "Usage\n");
    fprintf(stderr, "-----\n");
}

static inline bool parsell(char *s, long long *ret) {
    char *endpnt;
    *ret = strtoll(s, &endpnt, 10);
    if (*endpnt)
        return false;
    else
        return true;
}


static bool parse_option(int argc, char **argv) {
#define FLAG_FILL_BUFFER_WITH_ZERO 0
#define FLAG_MB 1
#define FLAG_KB 2
#define FLAG_BYTE 3
#define FLAG_TIMESTAMP_PROFILE 4
#define FLAG_TIMESTAMP_PROFILE_PATH 5
#define FLAG_GPU_FLAG 6
#define FLAG_USE_FIFO99 7
#define FLAG_RUN_COUNT 8
#define FLAG_RW_OFFSETS 9
#define FLAG_LEADER_ID 10

    static struct option opts [] = {
        {"fill-zero", no_argument, 0, FLAG_FILL_BUFFER_WITH_ZERO},
        {"mb", required_argument, 0, FLAG_MB},
        {"kb", required_argument, 0, FLAG_KB},
        {"byte", required_argument, 0, FLAG_BYTE},
        {"timestamp-profile", required_argument, 0, FLAG_TIMESTAMP_PROFILE},
        {"timestamp-profile-path", required_argument, 0, FLAG_TIMESTAMP_PROFILE_PATH},
        {"gpu-flag", required_argument, 0, FLAG_GPU_FLAG},
        {"use-fifo99", no_argument, 0, FLAG_USE_FIFO99},
        {"run-count", required_argument, 0, FLAG_RUN_COUNT},
        {"rw-offsets", required_argument, 0, FLAG_RW_OFFSETS},
        {"leader-id", required_argument, 0, FLAG_LEADER_ID},
        {0, 0, 0, 0}
    };

    program_option.fill_zero = false;
    program_option.use_ts_profile_file = false;
    program_option.use_ts_profile_str = false;
    program_option.use_rw_offsets = false;
    program_option.gpu_flag = 0;
    program_option.use_fifo99 = false;
    program_option.run_count = 1;
    program_option.leader_id = -1;

    int longopt_idx;
    int flag;
    char memsize_str[1024];

    int memsize_flag = -1;
    const char *memsize_flag_sig = "byte";
    long long memsize_coef = 1LL;
    while ((flag = getopt_long(argc, argv, "", opts, &longopt_idx)) != -1) {
        switch (flag) {
            case FLAG_FILL_BUFFER_WITH_ZERO:
                program_option.fill_zero = true;
                break;
            case FLAG_MB:
                strncpy(memsize_str, optarg, 1023);
                memsize_flag_sig = "mb";
                memsize_coef = 1024LL * 1024LL;
                memsize_flag = flag;
                break;
            case FLAG_KB:
                strncpy(memsize_str, optarg, 1023);
                memsize_flag_sig = "kb";
                memsize_coef = 1024LL;
                memsize_flag = flag;
                break;
            case FLAG_BYTE:
                strncpy(memsize_str, optarg, 1023);
                memsize_flag_sig = "mb";
                memsize_coef = 1LL;
                memsize_flag = flag;
                break;
            case FLAG_TIMESTAMP_PROFILE:
            {
                program_option.use_ts_profile_str = true;
                strncpy(program_option.ts_profile_str, optarg, 1023);
                break;
            }
            case FLAG_TIMESTAMP_PROFILE_PATH:
            {
                program_option.use_ts_profile_file = true;
                strncpy(program_option.ts_profile_path, optarg, 1023);
                break;
            }
            case FLAG_GPU_FLAG:
            {
                long long gpu_flag;
                if (!parsell(optarg, &gpu_flag)) {
                    fprintf(stderr, "--gpu-flag: %s is not a number\n", optarg);
                    return false;
                }
                program_option.gpu_flag = (unsigned long long)gpu_flag;
                break;
            }
            case FLAG_USE_FIFO99:
                program_option.use_fifo99 = true;
                break;
            case FLAG_RUN_COUNT:
            {
                long long run_count;
                if (!parsell(optarg, &run_count)) {
                    fprintf(stderr, "--run-count: %s is not a number\n", optarg);
                    return false;
                }
                program_option.run_count = (int)run_count;
                break;
            }
            case FLAG_RW_OFFSETS:
            {
                program_option.use_rw_offsets = true;
                strncpy(program_option.rw_offsets_str, optarg, 1023);
                break;
            }
            case FLAG_LEADER_ID:
            {
                long long li;
                if (!parsell(optarg, &li)) { 
                    fprintf(stderr, "--leader-id: %s is not a number\n", optarg);
                    return false;
                }
                program_option.leader_id = (int)li;
                break;
            }
            case '?':
                print_option();
                return false;
        }
    }
    if (memsize_flag == -1) {
        fprintf(stderr, "Memsize(byte, kb, mb) has not been specified!\n");
        return false;
    }

    if (program_option.gpu_flag == 0) {
        fprintf(stderr, "No gpu to use. abort.\n");
        return false;
    }


    int idx;
    long long memsize_size_t[MAX_GPU];
    ssize_t memsize_cnt = parse_colon_str(memsize_str, MAX_GPU, memsize_size_t);
    for (idx = 0; idx < MAX_GPU; idx++) {
        program_option.memsize[idx] = (size_t)memsize_size_t[idx];
    }

    if (memsize_cnt <= 0) {
        fprintf(stderr, "--%s: invalid format\n", memsize_flag_sig);
        return false;
    } else if (memsize_cnt == 1) {
        for (idx = 1; idx < MAX_GPU; idx++)
            program_option.memsize[idx] = program_option.memsize[0];
    }
    for (idx = 0; idx < MAX_GPU; idx++)
        program_option.memsize[idx] *= memsize_coef;
    program_option.mixed_mode = program_option.leader_id != -1;
    if (program_option.mixed_mode)
        fprintf(stderr, "MIXED MODE. leader-id=%d\n", program_option.leader_id);
    return true;
}

static ssize_t parse_colon_str(char *s, size_t max_cnt, long long *out) {
    char *p, *next;
    ssize_t cnt = 0;

    next = s - 1;
    do {
        p = next + 1;
        next = strstr(p, ":");
        if (next == NULL)
            next = p + strlen(p);

        char num_str[1024];
        strncpy(num_str, p, next - p);
        num_str[next - p] = 0;
        *(num_str + strlen(num_str)) = 0;

        char *endpnt;

        long long num;

        if (!*num_str) {
            num = 0;
        } else {
            num = strtoll(num_str, &endpnt, 10);
            if (*endpnt) {
                return -1;
            }
        }
        if (cnt < max_cnt)
            out[cnt++] = num;
        else
            return cnt;
    } while (*next);
    return cnt;
}

static bool get_memtest_shared_info(MemtestSharedInfo *shared_info, GpuLog *logger) {
    shared_info->randomize_buffer = !program_option.fill_zero;
    shared_info->logger = logger;
    memcpy(shared_info->memsize, program_option.memsize, sizeof(shared_info->memsize[0]) * MAX_GPU);

    memset(shared_info->gpu_ts_profile, 0, sizeof(shared_info->gpu_ts_profile[0]) * MAX_GPU);
    memset(shared_info->rw_offsets, 0, sizeof(shared_info->rw_offsets[0]) * MAX_GPU);
    if (program_option.use_ts_profile_str) {
        if (parse_colon_str(program_option.ts_profile_str, MAX_GPU, shared_info->gpu_ts_profile) == -1) {
            fprintf(stderr, "--timestamp-profile: not a valid format\n");
            return false;
        }
    } else {
        char *ts_profile_path = NULL;
        bool fp_success = false;

        if (program_option.use_ts_profile_file) {
            ts_profile_path = program_option.ts_profile_path;
        }

        if (ts_profile_path) {
            FILE *fp = fopen(ts_profile_path, "r");
            if (fp) {
                char buf[1024];
                size_t bytes_read;
                bytes_read = fread(buf, 1023, sizeof(char), fp);
                buf[bytes_read] = 0;
                fclose(fp);

                if (parse_colon_str(buf, MAX_GPU, shared_info->gpu_ts_profile) == -1) {
                    fprintf(stderr, "--timestamp-profile-path: the content of file has invalid format\n");
                    return false;
                }
                fp_success = true;
            } else if (errno != ENOENT) {
                fprintf(stderr, "Error occurred while reading from profile file '%s'", ts_profile_path);
                perror("");
                return false;
            }
        }

        if (!fp_success) {
            size_t gpu_cnt = average_timestamp_profile(MAX_GPU, shared_info->gpu_ts_profile);
            if (ts_profile_path) {
                FILE *fp = fopen(ts_profile_path, "w");
                if (!fp) {
                    fprintf(stderr, "Error occurred while writing to profile file '%s'", ts_profile_path);
                    perror("");
                    return false;
                }
                int idx;
                if (gpu_cnt > 0) {
                    fprintf(fp, "%lld", shared_info->gpu_ts_profile[0]);
                    for (idx = 1; idx < gpu_cnt; idx++) {
                        fprintf(fp, ":%lld", shared_info->gpu_ts_profile[idx]);
                    }
                }
                fclose(fp);
            }
        }
    }

    if (program_option.use_rw_offsets) {
        if (parse_colon_str(program_option.rw_offsets_str, MAX_GPU, shared_info->rw_offsets) == -1) {
            fprintf(stderr, "--rw-offsets: invalid format\n");
            return false;
        }
    }
    shared_info->gpu_flag = program_option.gpu_flag;
    shared_info->run_count = program_option.run_count;
    shared_info->leader_id = program_option.leader_id;
    shared_info->mixed_mode = program_option.mixed_mode;
    return true;
}

int main(int argc, char** argv) {
    if (!parse_option(argc, argv)) {
        return 1;
    }

    if (program_option.use_fifo99) {
        printf("Setting scheduler policy to FIFO 99...\n");
        struct sched_param sched_param;
        sched_param.sched_priority = 99;
        if (sched_setscheduler(0, SCHED_FIFO, &sched_param)) {
            printf("Failed to set scheduler policy. Are you root?\n");
            return 1;
        }
    }
    int gpu_idx = 0;
    unsigned long long gpu_iter;
    for (gpu_iter = program_option.gpu_flag; gpu_iter; gpu_iter >>= 1) {
        if (gpu_iter & 1) {
            fprintf(stderr, "GPU%d -> ", gpu_idx);
            fprintf(stderr, "Memory size: %zu Byte", program_option.memsize[gpu_idx]);
            if (program_option.memsize[gpu_idx] >= 1024 * 1024) {
                fprintf(stderr, "(%zu MB)", program_option.memsize[gpu_idx] / 1024 / 1024);
            } else if (program_option.memsize[gpu_idx] >= 1024) {
                fprintf(stderr, "(%zu KB)", program_option.memsize[gpu_idx] / 1024);
            }
            fprintf(stderr, "\n");
        }
        gpu_idx++;
    }

    GpuLog logger;
    MemtestSharedInfo shared_info;


    int gpu_cnt = evaluate_timestamp_profile(NULL);
    GpuLog_init(&logger, gpu_cnt);

    if (!get_memtest_shared_info(&shared_info, &logger)) {
        return 1;
    }
    
    Memtest memtest;
    Memtest_init(&memtest);
    Memtest_run(&memtest, &shared_info);
    Memtest_remove(&memtest);

    int gpu, idx;
    long long min_time = LLONG_MAX;
    for (gpu = 0; gpu < logger.gpu_cnt; gpu++) {
        size_t gpu_log_cnt = logger.gpu_log_cnt[gpu];
        for (idx = 0; idx < gpu_log_cnt; idx++) {
            GpuLogRecord *record = &logger.records_per_gpu[gpu][idx];
            if (min_time > record->queued_time) {
                min_time = record->queued_time;
            }
        }
    }
    for (gpu = 0; gpu < logger.gpu_cnt; gpu++) {
        size_t gpu_log_cnt = logger.gpu_log_cnt[gpu];
        for (idx = 0; idx < gpu_log_cnt; idx++) {
            GpuLogRecord *record = &logger.records_per_gpu[gpu][idx];
            printf("%d %s %lld %lld %lld\n", record->gpu_id, record->op_name, record->start_time - min_time, record->end_time - min_time, record->exec_time);
        }
    }
    return 0;
}
