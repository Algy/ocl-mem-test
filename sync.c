#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#include "sync.h"

void sb_init(SpinningBarrier *barrier, unsigned int nthread) {
    barrier->nthread = nthread;
    barrier->step = 0;
    barrier->nwait = 0;
}

bool sb_spin(SpinningBarrier *barrier) {
    unsigned int cur_step = barrier->step;
    unsigned int nthread_minus_one = barrier->nthread - 1;
    if (__sync_fetch_and_add(&barrier->nwait, 1) == nthread_minus_one) {
        barrier->nwait = 0; // XXX: maybe can use relaxed ordering here?
        __sync_fetch_and_add(&barrier->step, 1);
        return true;
    } else {
        while (barrier->step == cur_step);
        return false;
    }
}
