#!/usr/bin/env python

import subprocess
import os
import math

def parse_line(s):
    def consume_tabsp(idx):
        while idx < len(s):
            if s[idx] == ' ' or s[idx] == '\t':
                idx += 1
            else:
                break
        return idx
    result = {}
    idx = 0
    length = len(s)
    while idx < length:
        idx = consume_tabsp(idx)
        eq_idx = s.find("=", idx)
        if eq_idx == -1:
            raise ValueError("Invalid Format")
        key = s[idx:eq_idx]
        if s[eq_idx+1] != '[':
            raise ValueError("Invalid Format")
        cls_brk_idx = s.find(']', eq_idx + 2)
        if cls_brk_idx == -1:
            raise ValueError("Invalid Format")
        value = s[eq_idx + 2:cls_brk_idx].strip()
        idx = consume_tabsp(cls_brk_idx + 1)
        result[key] = value
    return result

def parse_opencl_profile_file(f):
    def generate_lines():
        line = f.readline()
        while line:
            yield line
            line = f.readline()


    line = f.readline()
    is_first_line = True
    for line in generate_lines():
        if line.startswith("#"):
            continue
        line = line.strip()
        if not line:
            continue
        # skip coloumn line
        if is_first_line:
            is_first_line = False
            continue
        # REFACTORME
        yield parse_line(line)

def exec_offtest():
    ret_code = subprocess.check_call(["./offtest"])
    if ret_code != 0:
        raise ValueError("offtest has exited with code %d"%ret_code)
    with open("opencl_profile_0.log", "r") as f0, \
         open("opencl_profile_1.log", "r") as f1:
        d0 = parse_opencl_profile_file(f0).next()
        d1 = parse_opencl_profile_file(f1).next()
        delta = float(int(d1["gpustarttimestamp"], 16) - 
                      int(d0["gpustarttimestamp"], 16)) * 1E-6
        return delta

class OfftestResult:
    def __init__(self, deltas):
        self.deltas = deltas
        self.mean = sum(deltas) / len(deltas)
        self.stdev = math.sqrt(sum((i - self.mean)**2 / (len(deltas) - 1) 
                                    for i in deltas))
    def show(self):
        print "Offset test result"
        print "----"
        print "Mean: %.2fms"%self.mean
        print "Stdev: %.2f"%self.stdev
        print "%s"%repr(self.deltas)

SAMPLE_N = 12
def gather_offtest():
    deltas = [exec_offtest() for _ in range(SAMPLE_N)]
    return OfftestResult(deltas)


OFFSET_FILE_PATH = "/tmp/.gpu-offset"
def get_device_offset():
    if os.path.isfile(OFFSET_FILE_PATH):
        with open(OFFSET_FILE_PATH) as f:
            return float(f.read())
    else:
        offt_res = gather_offtest()
        offt_res.show()
        with open(OFFSET_FILE_PATH, "w") as f:
            f.write(str(offt_res.mean))
        return offt_res.mean


def to_gnuplot_dat():
    device_offset = get_device_offset()
    with open("opencl_profile_0.log") as f0, \
         open("opencl_profile_1.log") as f1, \
         open("cldat.dat", "w") as outf:
        imd = [{}, {}]
        min_st_box = [None]
        def start_emit():
            pass
        def emit_data(gpu, method, st, ed):
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
            for gpu in range(2): 
                gpu_data = imd[gpu]
                for mtd, st_ed_list in gpu_data.items():
                    label = "GPU%d_%s"%(gpu, mtd)
                    for st, ed in st_ed_list:
                        st -= min_st
                        ed -= min_st
                        st *= 1E-6
                        ed *= 1E-6
                        if gpu != 0:
                            st -= device_offset
                            ed -= device_offset
                        outf.write("%d %s 0 %.2f %.2f %.2f\n"%(op_no, label, st, ed, ed - st))
                    op_no += 1

        def parse_data_of_line(gpu, d):
            st = int(d["gpustarttimestamp"], 16)
            ed = int(d["gpuendtimestamp"], 16)
            mtd = d["method"]
            emit_data(gpu, mtd, st, ed)

        start_emit()
        for d in parse_opencl_profile_file(f0):
            parse_data_of_line(0, d)
        for d in parse_opencl_profile_file(f1):
            parse_data_of_line(1, d)
        end_emit()

if __name__ == '__main__':
    to_gnuplot_dat()
