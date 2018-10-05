##
## calculate zeropoint for a given filter
import numpy as np
from astropy import units as u
from astropy import constants as const
import simmetis as sim
import glob

def calculate_zeropoint(filter_name,verbose=False):
	cmd = sim.UserCommands("../notebooks/metis_image_NQ.config")
	cmd["ATMO_USE_ATMO_BG"] = "yes"
	cmd["SCOPE_USE_MIRROR_BG"] = "yes"
	cmd["SIM_VERBOSE"]="no"
	cmd["FPA_QE"]="../data/TC_detector_METIS_NQ_no_losses.dat"
	cmd["INST_FILTER_TC"]="../data/TC_filter_"+filter_name+".dat"

	opt = sim.OpticalTrain(cmd)
	fpa = sim.Detector(cmd, small_fov=False)

	## generate a source with 0 mag
	lam, spec = sim.source.flat_spectrum(0, "../data/TC_filter_"+filter_name+".dat")
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

	f=[]
	bg=[]
	src=[]

	for filter in filters:
		filter=filter.split("TC_filter_")[1].split(".")[0]
		bg_counts, source_minus_bg_counts = calculate_zeropoint(filter)
		f.append(filter)
		bg.append(bg_counts)
		src.append(source_minus_bg_counts)
		print("{0:>15}: {1:.2E}    {2:.2E}".format(filter,bg_counts,source_minus_bg_counts))
