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


def run_individual_benchmark(func_obj, bench_data):
    start = time.perf_counter()
    result = func_obj(bench_data)
    end = time.perf_counter()
    elapsed_time_sec = end - start
    return result, elapsed_time_sec


def run_bench(array_size: int, num_trials: int = 3):
    """
    Compare pykokkos vs. NumPy performance
    """
    rng = np.random.default_rng(12345)
    data_np = np.float64(rng.random(array_size))
    data_cp = cp.asarray(data_np)
    view = pk.View([array_size], dtype=pk.double)
    view[:] = data_np
    timing_data = {"numpy": defaultdict(list),
                   "pykokkos": defaultdict(list),
                   "cupy": defaultdict(list)}
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
            result_pykokkos, time_pykokkos_sec = run_individual_benchmark(pk_func, view)
            result_cupy, time_cupy_sec = run_individual_benchmark(cp_func, data_cp)
            assert_allclose(result_pykokkos, result_numpy)
            assert_allclose(result_cupy.get(), result_numpy)
            timing_data["numpy"][func_name].append(time_numpy_sec)
            timing_data["pykokkos"][func_name].append(time_pykokkos_sec)
            timing_data["cupy"][func_name].append(time_cupy_sec)

    return timing_data, array_size


def plot_results(timing_data, array_size):
    fig, ax = plt.subplots(1, 1)
    x_labels = list(timing_data["numpy"].keys())
    width = 0.1
    x = np.arange(len(x_labels))

    for libname, offset in [("numpy", 0.0),
                            ("pykokkos", width),
                            ("cupy", width * 2)]:
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
    ax.set_title(f"Pykokkos unary ufunc performance vs. NumPy/CuPy for 1D array size of {array_size:.2E}",
                 fontsize=10)
    ax.legend()
    fig.savefig("pykokkos_vs_numpy_ufuncs.png",
                dpi=300)


if __name__ == "__main__":
    pk.set_default_space(pk.ExecutionSpace.Cuda)
    timing_data, array_size = run_bench(array_size=int(5e6),
                                        num_trials=4)
    plot_results(timing_data=timing_data,
                 array_size=array_size)
