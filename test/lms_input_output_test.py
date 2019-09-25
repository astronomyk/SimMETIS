## test script to  compare input / output flux

import simmetis as sim
from astropy import units as u
from astropy.io import fits
import numpy as np

lms = sim.spectro.LMS("../test_data/LINE_3D_013.fits", "../notebooks/metis_spectro_LMS.config")


input_cube = fits.open("../test_data/LINE_3D_013.fits")
input = np.sum(input_cube[0].data)
print("Total input flux [Jy]:")
print(input)
print()

result = lms.simulate('best', "PSF_SCAO_9mag_06seeing.fits", 6000., 200, plot=False)
result = lms.calibrate_flux(result)
pixel_scale = 8.2e-3
n_spectral_elements_ratio = np.shape(input_cube[0].data)[0]/np.shape(result)[0]
output = np.sum(result.data) * pixel_scale**2 * n_spectral_elements_ratio
print("Total output flux [Jy] (should be identical to input flux):")
print(output)
