#!/usr/bin/env python
import sys

from subprocess import check_output

NSAMPLE = 10

def percentage_d(num):
    return map(lambda p: int(num * p), [0.1, 0.25, 0.5, 0.75, 1])
    
def do_multi(offset, bytes_):
    src = check_output(["./memtest", "--byte", ":".join(map(str, bytes_)), "--rw-offsets", "0:%d"%offset, "--gpu-flag", "3", "--run-count", str(NSAMPLE)])
    lines = src.splitlines()

    gpu0_lines = [line.rsplit()[-1] for line in lines if line.startswith("0 ")]
    gpu1_lines = [line.rsplit()[-1] for line in lines if line.startswith("1 ")]

    gpu_h2d = [0, 0]
    gpu_d2h = [0, 0]

    for gpu_id, gpu_lines in enumerate([gpu0_lines, gpu1_lines]):
        gpu_h2d[gpu_id] = sum([float(line) / 1000. / 1000. for line in gpu_lines[0::3]]) / NSAMPLE
        gpu_d2h[gpu_id] = sum([float(line) / 1000. / 1000. for line in gpu_lines[2::3]]) / NSAMPLE
    
    yield ("H2D", offset, bytes_[0], bytes_[1]) + tuple(gpu_h2d)
    yield ("D2H", offset, bytes_[0], bytes_[1]) + tuple(gpu_d2h)

    src = check_output(["./memtest", "--byte", ":".join(map(str, bytes_)), "--rw-offsets", "0:%d"%offset, "--gpu-flag", "3", "--leader-id", "0", "--run-count", str(NSAMPLE)])
    lines = src.splitlines()

    gpu0_lines = [line.rsplit()[-1] for line in lines if line.startswith("0 ")]
    gpu1_lines = [line.rsplit()[-1] for line in lines if line.startswith("1 ")]

    gpu0_d2h = sum([float(line) / 1000. / 1000. for line in gpu0_lines[2::3]]) / NSAMPLE
    gpu1_h2d = sum([float(line) / 1000. / 1000. for line in gpu1_lines[0::3]]) / NSAMPLE

    yield ("MIXED", offset, bytes_[0], bytes_[1], gpu0_d2h, gpu1_h2d)


def do_single(gpu_id, byte):
    src = check_output(["./memtest", "--byte", str(byte), "--gpu-flag", "%d"%(1 << gpu_id), "--run-count", str(NSAMPLE)])
    lines = map(lambda x: x.rsplit()[-1], src.splitlines())
    time = sum([float(line) / 1000. / 1000. for line in lines[0::3]]) / NSAMPLE
    return ("D2H_SINGLE%d"%gpu_id, byte, time)


def multi_test_suite():
    for mb in range(50, 501, 50):
        byte = mb * 1024 * 1024
        for second_byte in percentage_d(byte):
            off = int(130 / 500. * mb * 1.1)
            offsets = set(range(-off, off, max(1, int(2 * off) / 20)))
            offsets.add(0) # zero always should be included
            for offset in sorted(offsets):
                for result in do_multi(offset, (byte, second_byte)):
                    sys.stdout.write("\t".join(map(str, result)))
                    sys.stdout.write("\n")
                    sys.stdout.flush()

def single_test_suite():
    mbs = set()
    for mb in range(50, 501, 50):
        for reduced_mb in percentage_d(mb):
            mbs.add(reduced_mb)

    for gpu in range(2):
        for mb in sorted(mbs):
            sys.stdout.write("\t".join(map(str, do_single(gpu, mb * 1024 * 1024))))
            sys.stdout.write("\n")
            sys.stdout.flush()

if __name__ == '__main__':
    # single_test_suite()
    multi_test_suite()
