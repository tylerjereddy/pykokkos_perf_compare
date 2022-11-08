import time
from collections import defaultdict

import numpy as np
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
    view = pk.View([array_size], dtype=pk.double)
    view[:] = data_np
    timing_data = {"numpy": defaultdict(list),
                   "pykokkos": defaultdict(list)}
    for func_name, pk_func, np_func in tqdm([("cos", pk.cos, np.cos),
                                             ("sin", pk.sin, np.sin),
                                            ]):
        for trial in tqdm(range(num_trials)):
            result_numpy, time_numpy_sec = run_individual_benchmark(np_func, data_np)
            result_pykokkos, time_pykokkos_sec = run_individual_benchmark(pk_func, view)
            assert_allclose(result_pykokkos, result_numpy)
            timing_data["numpy"][func_name].append(time_numpy_sec)
            timing_data["pykokkos"][func_name].append(time_pykokkos_sec)

    return timing_data, array_size


def plot_results(timing_data, array_size):
    fig, ax = plt.subplots(1, 1)
    x_labels = list(timing_data["numpy"].keys())
    x = np.arange(len(x_labels))
    heights_numpy_avg = []
    heights_numpy_std = []
    for func_name in x_labels:
        data = timing_data["numpy"][func_name]
        heights_numpy_avg.append(np.average(timing_data["numpy"][func_name]))
        heights_numpy_std.append(np.std(timing_data["numpy"][func_name]))

    num_trials = len(data)
    ax.bar(x=x,
           height=heights_numpy_avg,
           tick_label=x_labels,
           yerr=heights_numpy_std,
           capsize=20.0,
           label="NumPy",
           )
    ax.set_xlabel("Function compared")
    ax.set_ylabel(f"Avg +- std dev of time (s) for {num_trials} trials")
    ax.set_title(f"Pykokkos ufunc performance vs. NumPy for array size of {array_size:.2E}")
    ax.legend()
    fig.savefig("pykokkos_vs_numpy_ufuncs.png",
                dpi=300)


if __name__ == "__main__":
    timing_data, array_size = run_bench(array_size=int(5e6),
                                        num_trials=4)
    plot_results(timing_data=timing_data,
                 array_size=array_size)