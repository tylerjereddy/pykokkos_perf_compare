import time

import numpy as np
import pykokkos as pk

from perf_compare import run_individual_benchmark

def run_single_pykokkos_ufunc_bench(array_size=int(1e7)):
    # run a single pykokkos ufunc
    # for profiling with i.e., nsight-systems
    # the benchmark will include HTD and DTH transfers
    # for now at least
    rng = np.random.default_rng(12345)
    data_np = np.float64(rng.random(array_size))
    view = pk.View([array_size], dtype=pk.double)
    view[:] = data_np
    result_pykokkos, time_pykokkos_sec = run_individual_benchmark(pk.log, view)
    print("time_pykokkos_sec:", time_pykokkos_sec)


if __name__ == "__main__":
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    assert not pk.is_uvm_enabled()
    for trial in range(10):
        run_single_pykokkos_ufunc_bench()
