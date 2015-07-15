#!/usr/bin/env python
import sys

from subprocess import check_output

NSAMPLE = 6

def do_once(offset, mb):
    src = check_output(["./memtest", "--mb", str(mb), "--rw-offsets", "0:%d"%offset, "--gpu-flag", "3"])
    lines = src.splitlines()
    gpu0 = float(lines[0].rsplit()[-1]) / 1000. / 1000.
    gpu1 = float(lines[3].rsplit()[-1]) / 1000. / 1000.
    return (offset, mb, gpu0, gpu1)

def my_avg(lst, N):
    result = [lst[0][0], lst[0][1], 0, 0]
    for elem in lst:
        result[2] += elem[2] / float(N)
        result[3] += elem[3] / float(N)
    return result
        
def average(list_):
    arr = list(list_)
    N = len(arr)
    return my_avg(arr, N)


if __name__ == '__main__':
    for mb in range(50, 501, 50):
        off = int(130 / 500. * mb * 1.1)
        for offset in range(-off, off, max(1, int(2 * off) / 15)):
            sys.stdout.write("\t".join(map(str, average(do_once(offset, mb) for _ in range(NSAMPLE)))))
            sys.stdout.write("\n")
            sys.stdout.flush()
