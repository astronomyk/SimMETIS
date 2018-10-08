# Wishlist

### Future developments by Simcado classes

#### `OpticalTrain`
- include time-resolved jitter effects
- include sub-pixel shifts for spectro-spatial effects for Vortex coronagraphy
- instrument-internal emission (possibly requiring a different treatment of the global transmission/emission model) including **emission from entrance window**
- List of transmissive elements (e.g. multiple dichroics currently not supported), directly accessible e.g. via a dictionary
- Implement fixed ADCs (give residuals of the partial atmospheric dispersion correction as static input?)

#### `Detector`
- Chopping and nodding (including chop-difference residuals?)

### Other ideas

- Use PySynphot to replace spectral.py?

### More ideas from Ralf Siebenmorgen at CM06, May 2018

- simulating drift scans and compare it with chop/nod
- when possible please use naming of parameters as in the instrument dictionary of INS
- we shall have more discussions on artefacts by the system (sky, telescope, instrument) that shall be considered or not in the simulator
- I appreciate the use of the simulator for testing and verifying the science case performances (at least the major one and one use case per observing mode) also by MST
- When possible and applicable please benchmark the new against the old (idl) simulator


## Further things to think about

- implementation of cold stop(s) in METIS
- merge simcado and simmetis branches
- atmospheric disperion and partial correction in METIS
- more Aquarius noise properties (non-linear gain, correlated noise)