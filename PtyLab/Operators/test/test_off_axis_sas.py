import time

from PtyLab import easyInitialize
from PtyLab.Operators.off_axis_sas import propagate_off_axis_sas

try:
    import cupy as cp
    from cupyx.profiler import benchmark
except:
    pass



def load_data(path="example:simulation_cpm"):
    experimentalData, reconstruction, params, monitor, engine = easyInitialize(path)
    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    reconstruction.theta = (40, 0)
    
    return reconstruction, params

def test_off_axis_sas(test_device: str = "CPU", nruns: int = 10):
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

    if test_device == "GPU":
        if params.gpuSwitch:
            reconstruction._move_data_to_gpu()

            print("\nGPU Runs:")
            for i in range(nruns):
                t0 = time.time()
                _ = propagate_off_axis_sas(reconstruction.probe, params, reconstruction)
                t1 = time.time()
                print(f"Run {i}: {1e3 * (t1 - t0):.3f} ms")
        else:
            print("\nNo GPU hardware found, please set test_device = 'CPU' ")

    elif test_device == "CPU":
        params.gpuSwitch = False
        reconstruction._move_data_to_cpu()

        print("\nCPU Runs:")
        for i in range(nruns):
            t0 = time.time()
            _ = propagate_off_axis_sas(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(f"Run {i}: {1e3 * (t1 - t0):.3f} ms")
    else:
        raise SyntaxError("\nSet test_device = 'CPU' or 'GPU' ")


def benchmark_off_axis_sas(test_device: str = "CPU", nruns: int = 10):
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
    
    if test_device == "GPU":
        if params.gpuSwitch:
            reconstruction._move_data_to_gpu()
            run_propagator_func = lambda: propagate_off_axis_sas(reconstruction.probe, params, reconstruction)
            
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
            print("\nNo GPU hardware found, please set test_device = 'CPU'")
    
    elif test_device == "CPU":
        params.gpuSwitch = False
        reconstruction._move_data_to_cpu()
        
        print("\nCPU run times:")
        for i in range(nruns):
            t0 = time.time()
            propagate_off_axis_sas(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(f"Run {i}: {1e3 * (t1 - t0):.3f} ms")
    
    else:
        raise ValueError("Set test_device to 'CPU' or 'GPU'")


test_off_axis_sas(test_device="GPU", nruns=10)
benchmark_off_axis_sas(test_device="GPU", nruns=10)