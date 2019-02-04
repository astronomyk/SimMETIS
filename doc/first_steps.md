# Welcome to SimMETIS!
## First steps
We recommend to get familiar with the simulator by reading / re-executing and modifying the example notebooks that are part of the package. This way you learn both about the features of the simulator as well as input and output formats and you can quickly adapt these notebooks for your own purposes.

Additional information can be found in all SimMETIS objects. Within the convenient iPython environment you can both use tab completion as well as the `?`syntax, i.e.

#### Tab completion

`import simMETIS as sim`

`sim.` (then type the tabulator twice to see all methods of this object)

#### Question mark syntax
Type
`?sim.OpticalTrain`
to get an inline help of this particular object.

## Example notebooks
There are several example Jupyter (iPython) notebooks in the sub-directory `notebooks` that demonstrate the various observing modes of METIS. They are intended as templates that can easily be modified for your science case.

Notebooks can be opened, viewed and edited using the Python notebook viewer `jupyter`. They display comments, code as well as results, including plots in-line and are a popular way to share the results of calculations as they can easily be exported to PDF or LaTeX. To open the N band example notebook, `cd`into the notebook directory and execute
`jupyter notebook SimMETIS_IMG_N.ipynb`

### Imaging modes
Several notebooks demonstrate the usage of SimMETIS for the imaging modes of METIS.

`simulate_image.ipynb` shows how to run SimMETIS on an input image in a specified filter and is intended to serve as template for your own **imaging simulations**.

`SimMETIS_IMG_LM.ipynb` and `SimMETIS_IMG_N.ipynb` are example simulations for observations of a point source in L/M and N band filters and can be used to compute **point-source sensitivities**.

The other two notebooks are for test / development purposes and may serve illustrative purposes for you, but should not be used for your own simulations (or at least results should be treated with caution). `SimMETIS_IMG_N_zeropoint.ipynb` contains calculations to compare the zeropoint and background flux given by SimMETIS with a simple calculation and also with the values quoted in the METIS fluxes document (by Roy van Boekel). `SimMETIS_IMG_Q.ipynb` is a notebook for Q band imaging, but support for this band is still in development (the emission from the entrance window is not yet included in SimMETIS, but it is a significant contribution to the thermal background in the Q band).
The L/M band notebook is the simplest as it does not involve chopping or nodding, the N band notebook includes that and also a simple implementation of the N band camera's detector excess low-frequency noise (ELFN).

### Long-slit spectroscopy
(not yet implemented)

### L/M-band high resolution IFU (LMS)
The notebook `SimMETIS_LMS.ipynb` demonstrates the use of the L and M band high-resolution integral field unit of METIS (LMS). Important notes:

 - the input (spatial/spectral) resolution should be at least twice as good as the output resolution.
 - the input data cube must of course be a valid FITS file and have a valid WCS. We provide an example cube in the `test_data` directory. The `BUNIT` format has to be "Jy/Pixel" (case-insensitive). Any other value for this keyword (or if this keyword is missing) is interpreted as "Jy/arcsec2". The spectral axis can be either "FREQ", "VELO", or "WAVE".

## Point-Spread Functions
One of the most critical aspects of imaging simulations is a realistic simulation of the telescope and instrument's point spread function (PSF). In SimMETIS 0.1 two different PSFs are included:

 - `data/PSF_SCAO_9mag_06seeing.fits` contains the current best estimate of the PSF as corrected for atmospheric turbulence by the METIS-internal Single Conjugate Adaptive Optics (SCAO) system. It is an image cube with PSFs (including speckles) at a few representative wavelengths. SimMETIS will pick the one closest to your simulation wavelength and scale the PSF appropriately. This PSF is for a K=9 mag reference star under good seeing conditions (0.6" seeing in the V band). More PSFs can be downloaded from the [homepage of Markus Feldt](http://www.mpia.de/homes/feldt/METIS_AO/table.html) (MPIA).
 - For reference, `data/PSF_diffraction_limited_4.0_microns.fits` and `data/PSF_diffraction_limited_11.6_microns.fits` are also included. They represent PSFs at the theoretical maximum resolution. They are given for two representative wavelengths. SimMETIS picks the one that is closest to your simulation and interpolates the pixel scale for the wavelength range relevant in the actual filter.

 The default PSF that is used is the SCAO PSF. If, for example, you wanted to see what a diffraction-limited PSF would give you in the M band, then insert this line into your notebook after initialising the UserCommands object:
 
 `cmd["SCOPE_PSF_FILE"] = "PSF_diffraction_limited_4.0_microns.fits"`

## Configuration files
The main SimMETIS configuration files are

  - `simmetis/data/default.config` -- this is essentially a dictionary for all keywords that are allowed within SimMETIS. Keywords must not be removed from this file to ensure proper functioning of the software. It is preconfigured for METIS at the ELT and can essentially be ignored for normal use.
  - `notebooks/metis_image_LM.config`, `notebooks/metis_image_NQ.config` and `notebooks/metis_spectro_LMS.config` -- these are configuration files for specific modes of METIS, as the filename indicates. They set up the correct detector layout and noise pattern, pixel scale etc.

The simulation can be controlled by modifying the mode-specific configuration file, but we recommend not to do that, but instead over-write relevant parameters in the respective notebook. This is demonstrated in the example notebooks where, for example, the filter is explicity set using

`cmd["INST_FILTER_TC"] = "TC_filter_N2.dat"`

## Further documentation
SimMETIS is based on SimCADO which is [extensively documented](https://www.univie.ac.at/simcado/Home.html). For further information about the underlying classes, methods and structures, please refer to this documentation.
