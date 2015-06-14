#ifndef _SYNC_H
#define _SYNC_H
typedef struct {
    unsigned int nthread;
    volatile unsigned int step;
    volatile unsigned int nwait;
} SpinningBarrier;

void sb_init(SpinningBarrier *barrier, unsigned int nthread);
bool sb_spin(SpinningBarrier *barrier);


#endif
