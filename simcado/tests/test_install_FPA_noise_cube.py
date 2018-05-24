def install_FPA_noise_cube():
    import os
    import simcado as sim
    fname = os.path.join(sim.utils.__pkg_dir__, data, "FPA_nirspec_pca0.fits")
    if os.path.exists(fname):
        sim.detector.install_noise_cube(1)
        fname = os.path.join(sim.utils.__pkg_dir__, data, "FPA_noise.fits")
        return os.path.exists(fname)
    else:
        return False

def test_install_FPA():
    assert install_FPA_noise_cube() == True
