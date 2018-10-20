# First steps
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
There are four example Jupyter (iPython) notebooks in the sub-directory `notebooks`:

 - `SimMETIS_IMG_LM.ipynb` - `SimMETIS_IMG_N.ipynb` - `SimMETIS_IMG_N_zeropoint.ipynb`
 - `SimMETIS_IMG_Q.ipynb`

`SimMETIS_IMG_LM.ipynb` and `SimMETIS_IMG_N.ipynb` are example simulations for imaging observations in L/M and N band filters and are intended to serve as templates for your own simulations.

The other two notebooks are for test purposes and may serve illustrative purposes for you, but should not be used for your own simulations (or at least results should be treated with caution). `SimMETIS_IMG_N_zeropoint.ipynb` contains calculations to compare the zeropoint and background flux given by SimMETIS with a simple calculation and also with the values quoted in the METIS fluxes document (by Roy van Boekel). `SimMETIS_IMG_Q.ipynb` is a notebook for Q band imaging, but support for this band is still in development (in this band the entrance window, whose emissivity is not yet included in SimMETIS, contributes significantly to the background photons).

The L/M band notebook is the simplest as it does not involve chopping or nodding, the N band notebook includes that and also a simple implementation of the N band camera's detector excess low-frequency noise (ELFN).

## Point-Spread Functions
One of the most critical aspects of imaging simulations is a realistic simulation of the telescope and instrument's point spread function (PSF). In SimMETIS 0.1 two different PSFs are included:

 - `data/PSF_SCAO_9mag_06seeing.fits` contains the current best estimate of the PSF as corrected for atmospheric turbulence by the METIS-internal Single Conjugate Adaptive Optics (SCAO) system. It is an image cube with PSFs (including speckles) at a few representative wavelengths. SimMETIS will pick the one closest to your simulation wavelength and scale the PSF appropriately. This PSF is for a K=9 mag reference star under good seeing conditions (0.6" seeing in the V band). More PSFs can be downloaded from the [homepage of Markus Feldt](http://www.mpia.de/homes/feldt/METIS_AO/table.html) (MPIA).
 - For reference, `data/PSF_diffraction_limited_4.0_microns.fits` and `data/PSF_diffraction_limited_11.6_microns.fits` are also included. They represent PSFs at the theoretical maximum resolution. They are given for two representative wavelengths. SimMETIS picks the one that is closest to your simulation and interpolates the pixel scale for the wavelength range relevant in the actual filter.

 The default PSF that is used is the SCAO PSF. If, for example, you wanted to see what a diffraction-limited PSF would give you in the M band, then insert this line into your notebook after initialising the UserCommands object:
 
 `cmd["SCOPE_PSF_FILE"] = "PSF_diffraction_limited_4.0_microns.fits"`

## Configuration files
The main SimMETIS configuration files are

  - `simmetis/data/default.config` -- this is essentially a dictionary for all keywords that are allowed within SimMETIS. Keywords must not be removed from this file to ensure proper functioning of the software. It is preconfigured for METIS at the ELT and can essentially be ignored for normal use.
  - `notebooks/metis_image_LM.config` and `notebooks/metis_image_NQ.config` -- these configuration files for specific modes of METIS, as the filename indicates. They set up the correct detector layout and noise pattern, pixel scale etc.

The simulation can be controlled by modifying the mode-specific configuration file, but we recommend not to do that, but instead over-write relevant parameters in the respective notebook. This is demonstrated in the example notebooks where, for example, the filter is explicity set using

`cmd["INST_FILTER_TC"] = "TC_filter_N2.dat"`
