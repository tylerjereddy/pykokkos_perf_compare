import time

import numpy as np
import cupy as cp

from perf_compare import run_individual_benchmark

def run_single_cupy_ufunc_bench(array_size=int(1e7)):
    # run a single cupy ufunc
    # for profiling with i.e., nsight-systems
    # the benchmark will include HTD and DTH transfers
    # for now at least
    rng = np.random.default_rng(12345)
    data_np = np.float64(rng.random(array_size))
    result_cupy, time_cupy_sec = run_individual_benchmark(cp.log, data_np, "cupy_include_transfers")
    print("time_cupy_sec:", time_cupy_sec)


if __name__ == "__main__":
    for trial in range(10):
        run_single_cupy_ufunc_bench()
