# content of test_sample.py
def func(x):
    return x + 1


def test_right_answer():
    assert func(3) == 4


def run_simcado():
    import simcado as sim
    import os
    cmd = sim.UserCommands()
    cmd["OBS_EXPTIME"] = 3600
    cmd["OBS_NDIT"] = 1
    cmd["INST_FILTER_TC"] = "J"

    src = sim.source.source_1E4_Msun_cluster()
    opt = sim.OpticalTrain(cmd)
    fpa = sim.Detector(cmd)

    src.apply_optical_train(opt, fpa)
    fpa.read_out("my_output.fits")
    
    is_there = os.path.exists("my_output.fits")
    os.remove("my_output.fits")
    
    return is_there


def test_run_simcado():
    assert run_simcado() == True


def get_module_dir():
    import os
    import inspect
    __pkg_dir__ = os.path.dirname(inspect.getfile(inspect.currentframe()))
    return __pkg_dir__


def background_increases_consistently_with_exptime():
    """
    Run an empty source for exposure time: (1,2,4,8,16,32,64) mins
    If true the background level increases linearly and the stdev increases as sqrt(exptime)
    """
    
    import numpy as np
    import simcado as sim

    cmd = sim.UserCommands()
    cmd["OBS_REMOVE_CONST_BG"] = "no"

    opt = sim.OpticalTrain(cmd)
    fpa = sim.Detector(cmd)

    stats = []

    for t in [2**i for i in range(7)]:
        src = sim.source.empty_sky()
        src.apply_optical_train(opt, fpa)
        hdu = fpa.read_out(OBS_EXPTIME=60*t, FPA_LINEARITY_CURVE=None)
        im = hdu[0].data
        stats += [[t, np.sum(im), np.median(im), np.std(im)]]

    stats = np.array(stats)

    factors = stats / stats[0,:]
    bg_stats = [i == np.round(l**2) == np.round(j) == np.round(k) for i, j, k, l in factors] 

    return_val = np.all(bg_stats)
    if not return_val:
        print(factors)
    
    return return_val


def test_background_increases_consistently_with_exptime():
    assert background_increases_consistently_with_exptime() == True

