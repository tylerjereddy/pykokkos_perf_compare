import os
from typing import Optional
import time
from collections import defaultdict

import numpy as np
import cupy as cp
from numpy.testing import assert_allclose
import pykokkos as pk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_individual_benchmark(func_obj, bench_data, include_mem_transfers: Optional[str] = None):
    # we use include_mem_transfers mostly for CuPy at the moment, so that
    # it pays the time cost for HTD/DTH transfer like PyKokkos
    # does by default, to keep the performance comparisons
    # fair
    # TODO: how could we exclude the transfer timings
    # for the pykokkos ufuncs? A bit more tricky than CuPy
    # at the moment I think...
    if include_mem_transfers == "cupy_include_transfers":
        start = time.perf_counter()
        bench_data = cp.asarray(bench_data)
    elif include_mem_transfers == "cupy_exclude_transfers":
        bench_data = cp.asarray(bench_data)
        start = time.perf_counter()
    else:
        start = time.perf_counter()

    result = func_obj(bench_data)

    if include_mem_transfers == "cupy_include_transfers":
        result = result.get()
    elif include_mem_transfers == "cupy_exclude_transfers":
        # NOTE: cupyx.profiler.benchmark() for more granularity
        # of CPU vs. GPU timings

        # we synchronize the stream because there is otherwise some
        # danger that the CUDA kernel is still running
        # as far as the host is concerned
        cp.cuda.stream.get_current_stream().synchronize()

    end = time.perf_counter()

    if include_mem_transfers == "cupy_exclude_transfers":
        result = result.get()

    elapsed_time_sec = end - start
    return result, elapsed_time_sec


def run_bench(array_size: int, array_type: str, num_trials: int = 3):
    """
    Compare pykokkos vs. NumPy performance
    """
    if array_type == "float64":
        np_dtype = np.float64
        pk_dtype = pk.double
        rtol = 1e-7
    elif array_type == "float32":
        np_dtype = np.float32
        pk_dtype = pk.float
        rtol = 1e-6
    rng = np.random.default_rng(12345)
    data_np = np_dtype(rng.random(array_size))
    # we'll make this env variable mandatory:
    openmp_threads = os.environ["OMP_NUM_THREADS"]
    timing_data = {"numpy": defaultdict(list),
                   "pykokkos_cuda_no_uvm": defaultdict(list),
                   f"pykokkos_openmp_{openmp_threads}_threads": defaultdict(list),
                   "cupy_with_transfers": defaultdict(list),
                   "cupy_no_transfers": defaultdict(list)}
    for func_name, pk_func, np_func, cp_func in tqdm([("cos", pk.cos, np.cos, cp.cos),
                                             ("sin", pk.sin, np.sin, cp.sin),
                                             ("tan", pk.tan, np.tan, cp.tan),
                                             ("log", pk.log, np.log, cp.log),
                                             ("log2", pk.log2, np.log2, cp.log2),
                                             ("log10", pk.log10, np.log10, cp.log10),
                                             ("log1p", pk.log1p, np.log1p, cp.log1p),
                                             ("exp", pk.exp, np.exp, cp.exp),
                                             ("isnan", pk.isnan, np.isnan, cp.isnan),
                                             ("isfinite", pk.isfinite, np.isfinite, cp.isfinite),
                                            ]):
        for trial in tqdm(range(num_trials)):
            result_numpy, time_numpy_sec = run_individual_benchmark(np_func, data_np)
            # carefully set the views up after changing
            # execution spaces...
            pk.set_default_space(pk.ExecutionSpace.Cuda)
            assert not pk.is_uvm_enabled()
            view = pk.View([array_size], dtype=pk_dtype)
            view[:] = data_np
            result_pykokkos, time_pykokkos_sec = run_individual_benchmark(pk_func, view)
            pk.set_default_space(pk.ExecutionSpace.OpenMP)
            assert not pk.is_uvm_enabled()
            view = pk.View([array_size], dtype=pk_dtype)
            view[:] = data_np
            result_pykokkos_openmp, time_pykokkos_sec_openmp = run_individual_benchmark(pk_func, view)
            result_cupy, time_cupy_sec = run_individual_benchmark(cp_func, data_np, "cupy_include_transfers")
            result_cupy_no_xfer, time_cupy_sec_no_xfer = run_individual_benchmark(cp_func, data_np, "cupy_exclude_transfers")
            assert_allclose(result_pykokkos, result_numpy, rtol=rtol)
            assert_allclose(result_pykokkos_openmp, result_numpy, rtol=rtol)
            assert_allclose(result_cupy, result_numpy, rtol=rtol)
            assert_allclose(result_cupy_no_xfer, result_numpy, rtol=rtol)
            timing_data["numpy"][func_name].append(time_numpy_sec)
            timing_data["pykokkos_cuda_no_uvm"][func_name].append(time_pykokkos_sec)
            timing_data[f"pykokkos_openmp_{openmp_threads}_threads"][func_name].append(time_pykokkos_sec_openmp)
            timing_data["cupy_with_transfers"][func_name].append(time_cupy_sec)
            timing_data["cupy_no_transfers"][func_name].append(time_cupy_sec_no_xfer)

    return timing_data, array_size


def plot_results(timing_data, array_size, array_type):
    openmp_threads = os.environ["OMP_NUM_THREADS"]
    fig, ax = plt.subplots(1, 1)
    x_labels = list(timing_data["numpy"].keys())
    width = 0.1
    x = np.arange(len(x_labels))

    for libname, offset in [("numpy", 0.0),
                            ("pykokkos_cuda_no_uvm", width),
                            (f"pykokkos_openmp_{openmp_threads}_threads", width * 2),
                            ("cupy_with_transfers", width * 3),
                            ("cupy_no_transfers", width * 4)]:
        avg_time = []
        std_time = []
        for func_name in x_labels:
            data = timing_data[libname][func_name]
            avg_time.append(np.average(data[1:]))
            std_time.append(np.std(data[1:]))

        ax.bar(x=x + offset,
               height=avg_time,
               tick_label=x_labels,
               yerr=std_time,
               capsize=2.0,
               label=libname,
               width=0.1,
               )

    num_trials = len(data)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Function compared")
    ax.set_ylabel(f"Avg +- std dev of time (s) for {num_trials} trials [discard 1st trial]")
    ax.set_title((f"Pykokkos unary ufunc performance vs. NumPy/CuPy for 1D {array_type} array size of {array_size:.2E}\n"
                  "(benchmark source: https://github.com/tylerjereddy/pykokkos_perf_compare)"
                 ),
                 fontsize=10)
    ax.legend()
    fig.savefig(f"pykokkos_vs_numpy_ufuncs_{array_type}.png",
                dpi=300)


if __name__ == "__main__":
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    for array_type in ["float64", "float32"]:
        timing_data, array_size = run_bench(array_size=int(1e7),
                                            array_type=array_type,
                                            num_trials=4)
        plot_results(timing_data=timing_data,
                     array_size=array_size,
                     array_type=array_type)
