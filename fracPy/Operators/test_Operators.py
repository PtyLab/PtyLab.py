from numpy.testing import assert_allclose

from fracPy import Operators, easyInitialize


def test_object2detector():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize('example:simulation_cpm')
    params.gpuSwitch = True
    reconstruction._move_data_to_gpu()
    _doit(reconstruction, params)


    params.gpuSwitch = False
    reconstruction._move_data_to_cpu()
    _doit(reconstruction, params)


def _doit(reconstruction, params):
    for operator_name in Operators.Operators.forward_lookup_dictionary:
        params.propagatorType = operator_name
        reconstruction.esw = reconstruction.probe
        print('\n')
        import time

        for i in range(3):
            t0 = time.time()
            Operators.Operators.object2detector(reconstruction.esw, params, reconstruction)
            # out = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(operator_name, i, 1e3 * (t1 - t0), 'ms')


def test__propagate_fresnel():
    experimentalData, reconstruction, params, monitor, engine = easyInitialize('example:simulation_cpm')

    reconstruction.initializeObjectProbe()
    reconstruction.esw = 2
    for operator in [Operators.Operators.propagate_fresnel,
                      Operators.Operators.propagate_ASP,
                     Operators.Operators.propagate_scaledASP,
                     Operators.Operators.propagate_twoStepPolychrome,
                     Operators.Operators.propagate_scaledPolychromeASP]:
        params.gpuSwitch = True
        reconstruction._move_data_to_gpu()

        import time
        for i in range(3):
            t0 = time.time()
            out = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(i, 1e3*(t1-t0), 'ms')

        params.gpuSwitch = False
        reconstruction._move_data_to_cpu()

        import time
        for i in range(10):
            t0 = time.time()
            out = operator(reconstruction.probe, params, reconstruction)
            t1 = time.time()
            print(i, 1e3*(t1-t0), 'ms')

