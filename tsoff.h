#ifndef _TSOFF_H
#define _TSOFF_H

size_t evaluate_timestamp_profile(int *gpu_ts_profile_ret);
size_t average_timestamp_profile(size_t max_gpu, long long *profile_ret);
#endif
