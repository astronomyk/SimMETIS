# Requirements for the METIS data simulator
 - Simulate science observations with METIS in all modes in order to obtain performance estimates. Scientific priority: IMG (L/M+N), LMS (currently: no coronagraphic PSF simulation), long-slit spec. We need to simulate the following aspects of an observation:
  - atmospheric transmission + emission
  - thermal emission of the telescope (global emissivity or detailed model of telescope pupil?)
  - the whole optical train of the instrument (radiometric treatment of all transmissive + reflective elements + instrumental PSF)
  - dispersion effects (atmosphere + fixed ADC)
  - diffraction effects in the instrument (spectroscopic modes)
  - detector noise and QE
  - FITS file generation
  - convenience functions to output individual frames, half-cycle averaged unchopped images, chopped and nodded images
 - Tool for pipeline testing; requirements on the simulator:
  - FITS file structure
  - FITS header keywords
 - Tool for supporting the technical design (e.g. ADC performance); requirements on the simulator not yet clear; perhaps this may require extra output options (e.g. virtual detectors in other positions in the instrument)
 - Additionally, a new simulator development should also be well documented and come at least with a few example scripts (e.g. iPython notebooks) for each instrument / simulator mode that can easily be adapted by any scientist to run their own simulations.

## Scope
HCI PSF simulations will not be part of the METIS data simulator, but will be taken as a fixed input. They will be produced with dedicated simulators such as the one by Brunella Carlomagno (Liège) or HCIpy (Leiden).
