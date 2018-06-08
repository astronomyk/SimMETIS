# Development plan / Milestones
## 0.1 (goal: June 2018)

#### Goals
- Simcado with bug fixes to support simulations at 3-20 microns
- Configuration files for METIS L and N band imaging
- Transmission curves (sky emission / absorption, mirror reflectivity, entrance window, detector QE, dichroic transmission, filter curves) extended and/or adapted for METIS L and N bands
- iPython notebooks with example use cases for both LM and NQ band imaging
- output fluxes in L and N band imaging verified against Roy’s fluxes document

#### Work to be done
- N band
	- bg flux is only half as high as in Roy's document
		- correct emissivity/transmission of telescope
		- add emission from window and spiders
		- add atmospheric transmission and emission
	- check source photons
- verify correct fluxes and behaviour also in L band
- clean-up test notebooks
- clean-up and document example notebooks


## 0.2 (goal: July 2018)

#### Goals
- bug fixes
- properly implement cold stop(s) in METIS
- all filter curves
- M band
- Q band
- create pickled OpticalTrain objects for all METIS imaging
- observing modes for faster operation



## 0.3 (aim: September 2018, dependent on core SimCADO development)

- Long-slit mode with sensitivities verified against Roy’s fluxes document
- bug fixes
- updates for instrument configuration files

## 0.4 (aim: October 2018, dependent on core SimCADO development)

- IFU mode implemented with sensitivities verified against Roy’s fluxes document
- bug fixes
- updates for instrument configuration files

## 1.0 (aim: December 2018, dependent on Simcado development and humanpower available at NOVA)

- Aquarius noise generator included (including ELFN, non-linear gain curve, correlated noise)
- All METIS modes implemented, tested and verified against theoretical expectations and/or actual observations with similar instruments (e.g. VISIR, NaCO, SINFONI)
- Example notebooks available and reproducibly executable for all METIS modes