# Development plan / Milestones
## 0.1 (goal: end of July 2018)

#### Goals
- Simcado with bug fixes to support simulations at 3-20 microns
- Configuration files for METIS L and N band imaging
- Transmission curves (sky emission / absorption, mirror reflectivity, entrance window, detector QE, dichroic transmission, filter curves) extended and/or adapted for METIS L and N bands
- iPython notebooks with example use cases for both LM and NQ band imaging
- output fluxes in L and N band imaging verified against Roy’s fluxes document

#### Work to be done
- N band
	- integrate ELFN noise penalty according to Fig. 7 of Roy's fluxes technote (v 0.8) [Kieran]
	- check source photons (mag_to_photons function) [Oliver, Leo]
	- background flux
		- add atmospheric transmission and emission [all]
		- add emission from window and spiders [all]
- verify correct fluxes and behaviour also in L and M band [Enrico, Leo]
- clean-up test notebooks [Leo]
- clean-up and document example notebooks [Leo]


## 0.2 (goal: October 2018)
- Spectroscopy (perhaps first long-slit, then IFU) verified against Roy’s fluxes document


## Further things to do
#### Minor things
- all filter curves
- M band
- Q band
- create pickled OpticalTrain objects for all METIS imaging
- observing modes for faster operation

#### Things to think about
- first implementation of cold stop(s) in METIS
- merge simcado and simmetis branches
- atmospheric disperion and partial correction in METIS
- more Aquarius noise properties (non-linear gain, correlated noise)