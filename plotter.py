#!/usr/bin/env python

import subprocess
import sys
import os
import math

def to_gnuplot_dat(f, outf):
    imd = []
    min_st_box = [None]
    def start_emit():
        pass
    def emit_data(gpu, method, st, ed):
        if len(imd) <= gpu:
            for _ in range(gpu - len(imd) + 1):
                imd.append({})
        gpu_data = imd[gpu]
        d = gpu_data.get(method)
        if d is None:
            gpu_data[method] = d = []
        d.append((st, ed))
        min_st = min_st_box[0]
        if min_st is None or min_st > st:
            min_st_box[0] = st
    def end_emit():
        min_st = min_st_box[0]
        op_no = 1
        for gpu in range(len(imd)): 
            gpu_data = imd[gpu]
            for mtd, st_ed_list in gpu_data.items():
                label = "GPU%d_%s"%(gpu, mtd)
                for st, ed in st_ed_list:
                    st -= min_st
                    ed -= min_st
                    st *= 1E-6
                    ed *= 1E-6
                    outf.write("%d %s 0 %.3f %.3f %.3f\n"%(op_no, label, st, ed, ed - st))
                op_no += 1

    def parse_data_of_line(gpu, d):
        st = int(d["gpustarttimestamp"], 16)
        ed = int(d["gpuendtimestamp"], 16)
        mtd = d["method"]
        emit_data(gpu, mtd, st, ed)

    start_emit()
    for line in f.readlines():
        d = line.split()
        gpu_id = int(d[0])
        op_name = d[1]
        start_time = int(d[2])
        end_time = int(d[3])
        emit_data(gpu_id, op_name, start_time, end_time)
    end_emit()

if __name__ == '__main__':
    to_gnuplot_dat(sys.stdin, sys.stdout)
