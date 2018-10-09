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
There are three example Jupyter (iPython) notebooks in the sub-directory `notebooks` that are intended to be used for your simulations. Two are example simulations for imaging observations in L/M and N band filters and one is a test calculation to compare the zeropoint and background flux given by SimMETIS with a simple calculation and also with the values quoted in the METIS fluxes document (by Roy van Boekel).

 - `SimMETIS_IMG_LM.ipynb` - `SimMETIS_IMG_N.ipynb` - `SimMETIS_IMG_N_zeropoint.ipynb`The L/M notebook is the simplest as it does not involve chopping or nodding, the N band notebook includes that and also a simple implementation of the N band camera's detector excess low-frequency noise (ELFN). There is also a Q band notebook, but support for this band is still in development (in this band the entrance window contributes significantly to the background photons).