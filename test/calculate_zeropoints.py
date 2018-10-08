##
## calculate zeropoint for a given filter
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.io import ascii
import simmetis as sim
import glob

def calculate_zeropoint(filter_id,filter_path,verbose=False):
	cmd = sim.UserCommands("../notebooks/metis_image_generic.config")
	##
	## find out if we have the LM or NQ band camera
	filter_data=ascii.read(filter_path)
	if filter_data["col1"][0] < 6:
		cmd["FPA_QE"] = "../data/TC_detector_METIS_LM_no_losses.dat"
		cmd["SIM_DETECTOR_PIX_SCALE"] = 0.00525
		cmd["SCOPE_PSF_FILE"] = "PSF_4.0_microns.fits"
		cmd["FPA_QE"] = "TC_detector_METIS_LM.dat"
		cmd["FPA_CHIP_LAYOUT"] = "FPA_chip_layout.dat"
	else:
		cmd["FPA_QE"] = "../data/TC_detector_METIS_NQ_no_losses.dat"
		cmd["SIM_DETECTOR_PIX_SCALE"] = 0.01078
		cmd["OBS_EXPTIME"] = 1
		cmd["SCOPE_PSF_FILE"] = "PSF_11.6_microns.fits"
		cmd["FPA_CHIP_LAYOUT"] = "FPA_chip_layout_AQUARIUS.dat"
			
	cmd["INST_FILTER_TC"]="../data/TC_filter_"+filter_id+".dat"

	opt = sim.OpticalTrain(cmd)
	fpa = sim.Detector(cmd, small_fov=False)

	## generate a source with 0 mag
	lam, spec = sim.source.flat_spectrum(0, filter_path)
	src = sim.Source(lam=lam, spectra=np.array([spec]), ref=[0], x=[0], y=[0])
	src.apply_optical_train(opt, fpa)
	exptime=1
	##
	## noise-free image before applying Poisson noise
	photonflux = fpa.chips[0].array.T
	clean_image = photonflux * exptime
	##
	## detector image with Poisson noise
	hdu = fpa.read_out(OBS_EXPTIME=exptime)
	
	bg_counts = np.min(clean_image)
	source_minus_bg_counts = np.sum(clean_image - np.min(clean_image))
	
	if verbose:
		print("Background counts/s: {0:.2E}".format(bg_counts))
		print("Background-subtracted source counts/s: {0:.2E}".format(source_minus_bg_counts))
	return(bg_counts,source_minus_bg_counts)


def all_zeropoints():
	filters = glob.glob("../data/TC_filter_*.dat")
	
	for filter_path in filters:
		filename = filter_path.split("/")[2]
		filter_id = filter_path.split("TC_filter_")[1].split(".dat")[0]
		print(filter_id)
		bg_counts, source_minus_bg_counts = calculate_zeropoint(filter_id,filter_path)

		with open("zeropoints_simmetis.txt","a") as f:
			line="{0:>15}    {1:.2E}    {2:.2E}    {3}\n".format(filter_id,bg_counts,source_minus_bg_counts,filename)
			f.write(line)