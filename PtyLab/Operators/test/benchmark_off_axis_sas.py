import time

from PtyLab import easyInitialize
from PtyLab.Operators.off_axis_sas import propagate_sas
from PtyLab.utils.gpuUtils import check_gpu_availability

try:
    import cupy as cp  # type: ignore
    from cupyx.profiler import benchmark  # type: ignore
except ImportError:
    pass


def load_data(path="example:simulation_cpm"):
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(path)
    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    reconstruction.theta = (40, 0)
    reconstruction.z2 = reconstruction.zo * 1.3
    return reconstruction, params


def benchmark_runs(with_gpu=True, nruns: int = 10):
    """Checks if the off-axis SAS implementation works with some run-times.

    Parameters
    ----------
    test_device : str, optional
        Specify hardware either as "CPU" or "GPU", by default "CPU"
    nruns : int, optional
        No. of runs for each propagator
    """

    # load data
    reconstruction, params = load_data(path="example:simulation_cpm")
    reconstruction.pad_factor = 2  # can be modified by user.

    # # running on CPU

    # if check_gpu_availability(verbose=False):
    if with_gpu:
        if not check_gpu_availability(verbose=True):
            msg = "\nNo GPU hardware found, set with_gpu to False"
            raise AttributeError(msg)

        params.gpuSwitch = True
        reconstruction._move_data_to_gpu()

        def run_propagator_func():
            return propagate_sas(reconstruction.probe, params, reconstruction)

        # Warm-up run
        t0 = time.time()
        _ = run_propagator_func()
        t1 = time.time()

        print(f"\nGPU warm-up run time: {1e3 * (t1 - t0):.3f} ms")

        # Using CuPy's benchmark function
        print("\nGPU Benchmark Results:")
        result = benchmark(run_propagator_func, n_repeat=nruns)
        print(result)

        # Manual timing for comparison
        print("\nGPU run times:")
        for i in range(nruns):
            with cp.cuda.Stream() as stream:
                start = stream.record()
                run_propagator_func()
                end = stream.record()
                end.synchronize()
                elapsed = cp.cuda.get_elapsed_time(start, end)
                print(f"Run {i}: {elapsed:.3f} ms")
    else:
        print("\nNo GPU hardware found, benchmarking only for CPU ...")
        params.gpuSwitch = False
        reconstruction._move_data_to_cpu()

        print("\nCPU run times:")
        for i in range(nruns):
            t0 = time.time()
            propagate_sas(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(f"Run {i}: {1e3 * (t1 - t0):.3f} ms")


benchmark_runs(with_gpu=True, nruns=10)
