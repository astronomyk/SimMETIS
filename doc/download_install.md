# Download and installation instructions

## Contact and support
If you use SimMETIS, [we'd love to hear from you](simmetis@strw.leidenuniv.nl).

## Dependencies
In order to run SimMETIS, you need to have a working installation of Python 3 including the packages `numpy`, `scipy`, `astropy` and `matplotlib`, preferably in their latest versions. For special purposes, also `poppy` (Physical Optics Propagation in PYthon) in version 0.7.0 or higher as well as `wget` are required.

On the Mac, we recommend to install Python 3 using the package manager [Homebrew](https://brew.sh).

We further recommend using the interactive Python shell `iPython` as well as running your scripts in notebook mode using `Jupyter`.

Once SimMETIS is installed, you can find out which versions of these packages are used with

`simmetis.bug_report()`

Please include the output of this command when submitting bug reports. 

### Installation using pip
There are numerous ways to install python and its (many) optional packages. One of the easiest ways for normal users is the python package manager [pip](https://pip.pypa.io/en/stable/) and it is described here.

If you already have `pip`installed, you should upgrade it, before installing further packages using

`pip install --upgrade pip`

You can then install

`pip install numpy scipy astropy matplotlib ipython jupyter wget poppy`

or upgrade

`pip install --upgrade numpy scipy astropy matplotlib ipython jupyter wget poppy`

the dependencies for `SimMETIS`.

## Standard installation
Please download the .tar.gz bundle, unpack it at a directory of your choice (e.g. ~/SimMETIS) and install the package in your existing Python 3 installation by executing within the SimMETIS directory this command:

`pip install .`

SimMETIS is configured to run as-is. It has the following directories:
  - data -- contains METIS-specific instrument-configuration data (such as filters curves, detector QE curves, PSFs, ...)
  - doc -- contains this rudimentary documentation
  - notebooks -- contains a few example notebooks that should help for a quick start with the simulations
  - simmetis -- contains the actual simulator codebase (which is described on the SimCADO webpages)
  - test -- contains several scripts to verify that the outputs of SimMETIS agree with our back-of-the envelope calculations.
  - test_data -- test data used in the notebooks
  - 
## Developer installation
Alternatively, you can clone our github repository and run pip install in your local directory.
