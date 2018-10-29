"""
This module contains classes which control how a simulation is run

Summary
-------
UserCommands is essentially a dictionary that holds all the variables that
the user may wish to change. It also has some set variables like ``pix_res``
that can be accessed directly, instead of from the dictionary.

UserCommands is imported directly into the simmetis package and is accessible
from the main package - ``simmetis.UserCommands``

If ``UserCommands`` is called without any arguments, the default values for METIS
are used.

Classes
-------
``UserCommands(filename, default=<path_to_default>)``

Routines
--------

* ``dump_defaults(filename="./", selection="freq")``
* ``dump_chip_layout(dir="./")``


See Also
--------
Classes that require a ``UserCommands`` object directly include:

* ``Detector``
* ``OpticalTrain``


Notes
-----

References
----------

Examples
--------
By default ``UserCommands`` contains the parameters needed to generate the METIS
optical train:

    >>> my_cmds = simmetis.UserCommands()

To list the keywords that are available:

    >>> my_cmds.keys()
    ...


The UserCommands object also contains smaller dictionaries for each category of
keywords - e.g. for the keywords for the instrument:

    >>> my_cmds.inst
    ...


"""


import os
import shutil
import warnings
import logging

from collections import OrderedDict

import numpy as np
import astropy.io.ascii as ioascii    # ascii redefines builtin ascii().
#from astropy.io import fits  # unused

import simmetis as sim
from . import spectral as sc
from .utils import __pkg_dir__, atmospheric_refraction, find_file
from .psf import PSFCube

#__all__ = []
__all__ = ["UserCommands", "dump_defaults", "dump_chip_layout",
           "dump_mirror_config", "read_config", "update_config"]


class UserCommands(object):
    """
    An extended dictionary with the parameters needed for running a simulation

    Summary
    -------
    A :class:`.UserCommands` object contains a dictionary which holds all the keywords
    from the ``default.config`` file. It also has attributes which represent the
    frequently used variables, i.e. ``pix_res``, ``lam_bin_edges``, ``exptime``,
    etc

    ``<UserCommands>.cmds`` is a dictionary that holds all the variables
    the user may wish to change. It also has some set variables like
    ``<UserCommands>.pix_res`` that can be accessed directly, instead of from
    the dictionary.

    ``UserCommands`` is imported directly into the simmetis package and is
    accessable from the main package - ``simmetis.UserCommands``

    If UserCommands is called without any arguments, the default values for
    METIS and the E-ELT are used.


    Parameters
    ----------
    filename : str, optional
        path to the user's .config file

    Attributes
    ----------
    Internal dictionaries
    cmds : dict (collections.OrderedDict)
        the dictionary which holds all the keyword-value pairs needed for
        running a simualtion
    obs : dict (collections.OrderedDict)
        parameters about the observation
    sim : dict (collections.OrderedDict)
        parameters about the simualtion
    atmo : dict (collections.OrderedDict)
        parameters about the atmosphere
    scope : dict (collections.OrderedDict)
        parameters about the telescope
    inst : dict (collections.OrderedDict)
        parameters about the instrument
    fpa : dict (collections.OrderedDict)
        parameters about the detector array (FPA - Focal Plane Array)
    hxrg : dict (collections.OrderedDict)
        parameters about the chip noise (HxRG - HAWAII 4RG chip series)

    Attributes pertaining to the purely spectral data sets (e.g. transmission
    curves, stellar spectra, etc)
    lam : np.ndarray
        a vector containing the centres of the wavelength bins used when
        resampling the spectra or transmission curves
    lam_res : float
        [um] the resolution of the ``lam``

    Attributes pertaining to the binning in spectral space for which different
    PSFs need to be used
    lam_psf_res : float
        [um] the spectal "distance" between layers - i.e. width of the bins
    lam_bin_edges : array-like
        [um] the edge of the spectral bin for each layer
    lam_bin_centers : array-like
        [um] the centres of the spectral bin for each layer

    Attributes pertaining to the binning in the spatial plane
    pix_res : float
        [arcsec] the internal (oversampled) spatial resolution of the simulation
    fpa_res : float
        [arcsec] the field of view of the individual pixels on the detector

    General attributes
    verbose : bool
        Flag for printing intermediate results to the screen (default=True)
    exptime : float
        [s] exposure time of a single DIT
    diameter : float
        [m] outer diamter of the primary aperture (i.e. M1)
    area : float
        [m^2] effective area of the primary aperture (i.e. M1)
    filter : str
        [BVRIzYJHKKs,user] filter used for the observation

    Methods
    -------
    update(new_dict)
        updates the current ``UserCommands`` object from another dict-like object
    keys()
        returns the keys in the ``UserCommands.cmds`` dictionary
    values()
        returns the values in the ``UserCommands.cmds`` dictionary

    Raises
    ------

    See Also
    --------
    Detector, OpticalTrain

    Notes
    -----

    References
    ----------

    Examples
    --------
    By default ``UserCommands`` contains the parameters needed to generate the
    METIS optical train:

        >>> import simmetis
        >>> my_cmds = simmetis.UserCommands()


    To list the keywords that are available:

        >>> my_cmds.keys()
        ...


    The ``UserCommands`` object also contains smaller dictionaries for each category
    of keywords - e.g. for the keywords describing the instrument:

        >>> my_cmds.inst
        ...


    """


    def __init__(self, filename=None):

        """
        Create an extended dictionary of simulation parameters

        Parameters
        ----------
        filename : str, optional
            path to the user's .config file

        """

        logging.info("UserCommands object created")

        self.pkg_dir = __pkg_dir__
        default = os.path.join(self.pkg_dir, "data", "default.config")

        # read in the default keywords
        self.cmds = read_config(default)

        # read in the users wishes
        if filename is not None:
            self.cmds.update(read_config(filename))

        # add the instrument-specific data directory to the package
        # search path
        if sim.__search_path__[1] != self.cmds['SIM_DATA_DIR']:
            sim.__search_path__.insert(1, self.cmds['SIM_DATA_DIR'])

        # into python None values
        self._convert_none()
        self._find_files()
        self._default_data()

        self.cmds["CONFIG_USER"] = filename
        self.cmds["CONFIG_DEFAULT"] = default

        if self.cmds["SIM_PSF_OVERSAMPLE"] == "yes":
            self.cmds["PSF_MODE"] = "oversample"
        else:
            self.cmds["PSF_MODE"] = "linear_interp"

        # Subcategories of parameters, filled later by _split_categories
        # in _update_attributes
        self.obs = None
        self.sim = None
        self.atmo = None
        self.scope = None
        self.inst = None
        self.fpa = None
        self.hxrg = None

        # update the UserCommand "special" attributes
        self._update_attributes()

        if self.verbose and filename is not None:
            print("Read in parameters from " + filename)
            logging.debug("Read in parameters from " + filename)



    def update(self, new_dict):
        """
        Update multiple entries of a ``UserCommands`` dictionary

        ``update(new_dict)`` takes either a normal python ``dict`` object or a
        ``UserCommands`` object. Only keywords that match those in the
        ``UserCommands`` object will be updated. Misspelled keywords raise an
        error.

        To update single items in the dictionary, it is recommended to simply
        call the key and update the value - i.e ``<UserCommands>[key] = value``.

        Parameters
        ----------
        new_dict : dict, ``UserCommands``


        Raises
        ------
        KeyError
            If a parameter is not found in ``self.cmds``.

        See Also
        --------
        UserCommands

        Notes
        -----

        Examples
        --------
        View the default commands

            >>> import simmetis
            >>> my_cmds = simmetis.UserCommands()
            >>> print(my_cmds.cmds)


        Change a single command

            >>> my_cmds["OBS_EXPTIME"] = 60


        Change a series of commands at once

            >>> new_cmds = {"OBS_EXPTIME" : 60 , "OBS_NDIT" : 10}
            >>> my_cmds.update(new_cmds)


        """

        if isinstance(new_dict, UserCommands):
            tmpcmds = new_dict.cmds
        elif isinstance(new_dict, dict):
            tmpcmds = new_dict
        else:
            raise ValueError("Cannot update with type: "+type(new_dict))

        for thekey in tmpcmds:
            if not thekey in self.cmds:
                raise KeyError("Unknown parameter " + thekey)
            else:
                self.cmds[thekey] = tmpcmds[thekey]

        self._find_files()
        self._default_data()
        self._update_attributes()


    def keys(self):
        """
        Return the keys in the `UserCommands.cmds` dictionary
        """
        return self.cmds.keys()


    def values(self):
        """
        Return the values in the `UserCommands.cmds` dictionary
        """
        return self.cmds.values()


    def writeto(self, filename="commands.config"):
        """
        Write all the key-value commands to an ASCII file on disk

        Parameters
        ----------
        filename : str
            file path for where the file should be saved
        """
        outstr = ""
        for group in (self.obs, self.sim,
                      self.atmo, self.scope, self.inst,
                      self.fpa, self.hxrg):
            for key in group:
                val = self[key]
                if key == "FPA_CHIP_LAYOUT" and "\n" in val:
                    val = "small"
                outstr += key.ljust(25)+"  "+str(val) + "\n"
            outstr += "\n"
        with open(filename, "w") as fd1:
            fd1.write(outstr)


    def _convert_none(self):
        """
        Turn all string "none" or "None" values into python ``None`` values
        """
        for key in self.cmds:
            value = self.cmds[key]
            if isinstance(value, str) and value.lower() == "none":
                self.cmds[key] = None


    def _find_files(self):
        """
        Checks for files in the package search path

        The search path comprises the directories:
            "./", SIM_DATA_DIR, <pkg_dir>, <pkg_dir>/data
        """

        for key in self.cmds:
            if key == "OBS_OUTPUT_DIR":       # need not exist
                continue

            keyval = self.cmds[key]

            # not a string: not a filename
            if not isinstance(keyval, str):
                continue

            # If string has no extension, assume it's not a file name.
            # This is a strong assumption, but we need to guard from
            # looking for "yes", "no", "none", "scao", etc.
            # TODO Can we have a list of reserved keywords?
            if "." in keyval and len(keyval.split(".")[-1]) > 1:
                # look for the file
                fname = find_file(keyval, silent=True)
                if fname is None:
                    warnings.warn("Keyword "+key+" path doesn't exist: "
                                  + keyval)
                else:
                    self.cmds[key] = fname


    def _default_data(self):
        """
        Input system-specific path names for the default package data files
        """

        if isinstance(self.cmds["SCOPE_PSF_FILE"], str):
            if self.cmds["SCOPE_PSF_FILE"].lower() in ["ltao"]:
                self.cmds["SCOPE_PSF_FILE"] = find_file("PSF_LTAO.fits")
            elif self.cmds["SCOPE_PSF_FILE"].lower() in ("default", "scao"):
                self.cmds["SCOPE_PSF_FILE"] = find_file("PSF_SCAO.fits")
                self.cmds["INST_USE_AO_MIRROR_BG"] = "no"
            elif self.cmds["SCOPE_PSF_FILE"].lower() in ("mcao", "maory"):
                print("Unfortunately SimMETIS doesn't yet have a MCAO PSF")
                print("Using the SCAO PSF instead")
                self.cmds["SCOPE_PSF_FILE"] = find_file("PSF_SCAO.fits")
            elif self.cmds["SCOPE_PSF_FILE"].lower() in ("poppy", "ideal"):
                self.cmds["SCOPE_PSF_FILE"] = find_file("PSF_POPPY.fits")
            elif (find_file(self.cmds["SCOPE_PSF_FILE"]) is None):
                raise ValueError("Cannot recognise PSF file name: " +
                                 self.cmds["SCOPE_PSF_FILE"])
        elif isinstance(self.cmds["SCOPE_PSF_FILE"], PSFCube):
            pass

        elif self.cmds["SCOPE_PSF_FILE"] is None:
            warnings.warn("SCOPE_PSF_FILE is None - Generating PSF from OBS_SEEING")
            logging.debug("SCOPE_PSF_FILE is None - Generating PSF from OBS_SEEING")

        else:
            raise ValueError("Cannot recognise SCOPE_PSF_FILE: " +
                             self.cmds["SCOPE_PSF_FILE"])


        if self.cmds["INST_MIRROR_TC"] == "default":
            self.cmds["INST_MIRROR_TC"] = self.cmds["SCOPE_M1_TC"]

        if self.cmds["INST_MIRROR_AO_TC"] == "default":
            self.cmds["INST_MIRROR_AO_TC"] = self.cmds["INST_MIRROR_TC"]


        # which detector chip to use
        if self.cmds["FPA_CHIP_LAYOUT"] in (None, "none", "default", "full"):
            self.cmds["FPA_CHIP_LAYOUT"] = \
                find_file("FPA_chip_layout.dat")
        elif self.cmds["FPA_CHIP_LAYOUT"].lower() == "small":
            self.cmds["FPA_CHIP_LAYOUT"] = \
                find_file("FPA_chip_layout_small.dat")
        elif self.cmds["FPA_CHIP_LAYOUT"].lower() == "tiny":
            self.cmds["FPA_CHIP_LAYOUT"] = \
                find_file("FPA_chip_layout_tiny.dat")
        elif self.cmds["FPA_CHIP_LAYOUT"].lower() in ("centre", "central",
                                                      "middle", "center"):
            self.cmds["FPA_CHIP_LAYOUT"] = \
                find_file("FPA_chip_layout_centre.dat")



    def _update_attributes(self):
        """
        Update the UserCommand convenience attributes
        """

        if self.cmds["SCOPE_MIRROR_LIST"] is not None:
            self.mirrors_telescope = ioascii.read(self.cmds["SCOPE_MIRROR_LIST"])
        else:
            raise ValueError("SCOPE_MIRROR_LIST = " + \
                                                self.cmds["SCOPE_MIRROR_LIST"])

        if self.cmds["INST_MIRROR_AO_LIST"] is not None:
            self.mirrors_ao = ioascii.read(self.cmds["INST_MIRROR_AO_LIST"])
        else:
            self.mirrors_ao = ioascii.read("""
            #Mirror     Outer   Inner   Temp
            M0          0.     0.      -273
            """)

        i = np.where(self.mirrors_telescope["Mirror"] == "M1")[0][0]
        self.diameter = self.mirrors_telescope["Outer"][i]
        self.area = np.pi / 4 * (self.diameter**2 - \
                                 self.mirrors_telescope["Inner"][i]**2)


        # Check for a filter curve file or a standard broadband name
        if isinstance(self.cmds["INST_FILTER_TC"], str):

            fname = find_file(self.cmds["INST_FILTER_TC"])

            if fname is None:
                # try the name of the filter
                tryname = "TC_filter_" + self.cmds["INST_FILTER_TC"] + ".dat"
                fname = find_file(tryname)
                if fname is None:
                    raise ValueError("Filter " + self.cmds["INST_FILTER_TC"] +
                                     "could not be found")

            self.cmds["INST_FILTER_TC"] = fname

        self.fpa_res = self.cmds["SIM_DETECTOR_PIX_SCALE"]
        self.pix_res = self.fpa_res / self.cmds["SIM_OVERSAMPLING"]

        # If OBS_ZENITH_DIST is specified it overrides ATMO_AIRMASS,
        if self.cmds["OBS_ZENITH_DIST"] is not None:
            self.cmds["ATMO_AIRMASS"] = \
                sim.utils.zendist2airmass(self.cmds["OBS_ZENITH_DIST"])

        # if SIM_USE_FILTER_LAM is true, then use the filter curve to set the
        # wavelength boundaries where the filter is < SIM_FILTER_THRESHOLD

        if self.cmds["SIM_USE_FILTER_LAM"].lower() == "yes":
            if isinstance(self.cmds["INST_FILTER_TC"], str):
                tc_filt = sc.TransmissionCurve(filename=self.cmds['INST_FILTER_TC'])
            else:
                tc_filt = self.cmds["INST_FILTER_TC"]
            mask = np.where(tc_filt.val > self.cmds["SIM_FILTER_THRESHOLD"])[0]
            imin = np.max((mask[0] - 1, 0))
            imax = np.min((mask[-1] + 1, len(tc_filt.lam) - 1))
            lam_min, lam_max = tc_filt.lam[imin], tc_filt.lam[imax]
        else:
            lam_min = self.cmds["SIM_LAM_MIN"]
            lam_max = self.cmds["SIM_LAM_MAX"]

        self.lam_res = self.cmds["SIM_LAM_TC_BIN_WIDTH"]
        self.lam = np.arange(lam_min, lam_max + 1E-7, self.lam_res)

        #self.lam_psf_res = self.cmds["SIM_LAM_PSF_BIN_WIDTH"]
        #self.lam_bin_edges = np.arange(lam_min,
        #                               lam_max + self.lam_psf_res + 1E-7,
        #                               self.lam_psf_res)
        # make lam_bin_edges according to how great the ADC offsets are
        self.lam_bin_edges = self._get_lam_bin_edges(lam_min, lam_max)
        self.lam_bin_centers = 0.5 * (self.lam_bin_edges[1:] + \
                                      self.lam_bin_edges[:-1])

        self.exptime = self.cmds["OBS_EXPTIME"]


        # replace 'none', 'None' with None
        self._convert_none()

        self.verbose = (self.cmds["SIM_VERBOSE"] == "yes")
        self._get_total_wfe()

        self._split_categories()


    def _get_total_wfe(self):
        """
        Gets the total wave front error from the table in INST_WFE_LIST
        """

        if self.cmds["INST_WFE"] is not None:
            if isinstance(self.cmds["INST_WFE"], str):
                wfe_list = ioascii.read(self.cmds["INST_WFE"])
                wfe = wfe_list[wfe_list.colnames[0]]
                num = wfe_list[wfe_list.colnames[1]]
            elif isinstance(self.cmds["INST_WFE"], (float, int)):
                wfe, num = float(self.cmds["INST_WFE"]), 1

            tot_wfe = np.sqrt(np.sum(num * wfe**2))
        else:
            tot_wfe = 0

        self.cmds["INST_TOTAL_WFE"] = tot_wfe


    def _get_lam_bin_edges(self, lam_min, lam_max):
        """
        Generates an array with the bin edges of the layers in spectral space

        Parameters
        ----------
        lam_min, lam_max : float
            [um] the minimum and maximum wavelengths of the filter range

        Notes
        -------
        Atmospheric diffraction causes blurring in an image. To model this
        effect the spectra from a ``Source`` object are cut into bins based on
        how far the photons at different wavelength are diffracted from the
        image center. The threshold for defining a new layer based on the how
        far a certain bin will move is given by ``SIM_ADC_SHIFT_THRESHOLD``. The
        default value is 1 pixel.

        The PSF also causes blurring as it spreads out over a bandpass. This
        also needed to be taken into account
        """
        if self.cmds["SIM_VERBOSE"] == "yes":
            print("Determining lam_bin_edges")
        logging.debug("[UserCommands] Determining lam_bin_edges")

        effectiveness = self.cmds["INST_ADC_PERFORMANCE"] / 100.

        # This is redundant because also need to look at the PSF width
        #if effectiveness == 1.:
        #    lam_bin_edges = np.array([lam_min, lam_max])
        #    return lam_bin_edges

        shift_threshold = self.cmds["SIM_ADC_SHIFT_THRESHOLD"]

        ## get the angle shift for each slice
        lam = np.arange(lam_min, lam_max + 1E-7, 0.001)
        zenith_distance = sim.utils.airmass2zendist(self.cmds["ATMO_AIRMASS"])
        angle_shift = atmospheric_refraction(lam,
                                             zenith_distance,
                                             self.cmds["ATMO_TEMPERATURE"],
                                             self.cmds["ATMO_REL_HUMIDITY"],
                                             self.cmds["ATMO_PRESSURE"],
                                             self.cmds["SCOPE_LATITUDE"],
                                             self.cmds["SCOPE_ALTITUDE"])

        ## convert angle shift into number of pixels
        ## pixel shifts are defined with respect to last slice
        rel_shift = (angle_shift - angle_shift[-1]) / self.pix_res
        rel_shift *= (1. - effectiveness)
        if np.max(np.abs(rel_shift)) > 1000:
            raise ValueError("Pixel shifts too great (>1000), check units")

        ## Rotate by the paralytic angle
        int_shift = np.array(rel_shift / shift_threshold, dtype=np.int)
        idx = [np.where(int_shift == i)[0][0]
               for i in np.unique(int_shift)[::-1]]
        lam_bin_edges_adc = np.array(lam[idx + [len(lam)-1]])

        # Now check to see if the PSF blurring is the controlling factor. If so,
        # take the lam_bin_edges for the PSF blurring

        diam = self.diameter
        d_ang = self.pix_res * shift_threshold

        lam_bin_edges_psf = [lam_min]
        ang0 = (lam_min*1E-6) / diam * 1.22*53*3600

        i = 1
        while lam_bin_edges_psf[-1] < lam_max:
            lam_bin_edges_psf += [(ang0 + d_ang*i) * diam / (1.22*53*3600) * 1E6]
            i += 1
            if i > 1000:
                raise ValueError("lam_bin_edges needs >1000 values")
        lam_bin_edges_psf[-1] = lam_max

        lam_bin_edges = np.unique(np.concatenate(
            (np.round(lam_bin_edges_psf, 3),
             np.round(lam_bin_edges_adc, 3))))

        if self.cmds["SIM_VERBOSE"] == "yes":
            print("PSF edges were", np.round(lam_bin_edges_psf, 3))
            print("ADC edges were", np.round(lam_bin_edges_adc, 3))
            print("All edges were", np.round(lam_bin_edges, 3))
        logging.debug("[UserCommands] lam_bin_edges found")

        return lam_bin_edges


    def _split_categories(self):
        """
        Generate smaller category-specific dictionaries
        """
        self.obs   = {i:self.cmds[i] for i in self.cmds.keys() if "OBS" in i}
        self.sim   = {i:self.cmds[i] for i in self.cmds.keys() if "SIM" in i}
        self.atmo  = {i:self.cmds[i] for i in self.cmds.keys() if "ATMO" in i}
        self.scope = {i:self.cmds[i] for i in self.cmds.keys() if "SCOPE" in i}
        self.inst  = {i:self.cmds[i] for i in self.cmds.keys() if "INST" in i}
        self.fpa   = {i:self.cmds[i] for i in self.cmds.keys() if "FPA" in i}
        self.hxrg  = {i:self.cmds[i] for i in self.cmds.keys() if "HXRG" in i}


    def __str__(self):
        if self.cmds["CONFIG_USER"] is not None:
            return "A dictionary of commands compiled from " + \
                                                        self.cmds["CONFIG_USER"]
        return "A dictionary of default commands"

    def __iter__(self):
        return self.cmds.__iter__()

    def __getitem__(self, key):
        return self.cmds[key]

    def __setitem__(self, key, val):
        if key not in self.cmds.keys():
            raise ValueError(key+" not in UserCommands.keys()")

        self.cmds[key] = val
        self._find_files()
        self._default_data()
        self._update_attributes()


    ### Add to the update that all the cmds.variable are updated when
    ### the dicts are updated


def dump_defaults(filename=None, selection="freq"):
    ## OC, 2016-08-11: changed parameter from 'type' to 'selection' as
    ##    'type' redefines built-in
    """
    Dump the frequent.config file to a path specified by the user

    Parameters
    ----------
    filename : str, optional
        path or filename where the .config file is to be saved
    selection : str, optional
        ["freq", "all"] amount of keywords to save. "freq" only prints the most
        frequently used keywords. "all" prints all of them
    """

    if "freq" in selection.lower():
        fname = "frequent.config"
    elif "all" in selection.lower():
        fname = "default.config"

    if filename is None:
        gname = os.path.join(__pkg_dir__, "data", fname)
        f = open(gname, "r")
        print(f.read())
        f.close()
        return None
    else:
        path, gname = os.path.split(filename)
        if path == "":
            path = "."

        if gname == "":
            gname = fname
        shutil.copy(os.path.join(__pkg_dir__, "data", fname),
                    os.path.join(path, gname))


def dump_chip_layout(path=None):
    ## OC, 2016-08-11: changed parameter from 'dir' (redefines built-in)
    """
    Dump the FPA_chip_layout.dat file to a path specified by the user

    Parameters
    ----------
    path : str, optional
        path where the chip layout file is to be saved
    """
    fname = find_file("FPA_chip_layout.dat")

    if path is None:
        f = open(fname, "r")
        print(f.read())
        f.close()
    else:
        path = os.path.dirname(path)
        shutil.copy(fname, path)
        logging.debug("Printed chip layout to file: "+path+"/"+fname)


def dump_mirror_config(path=None, what="scope"):
    """
    Dump the EC_mirrors_scope.tbl or the EC_mirrors_ao.tbl to disk

    Parameters
    ----------
    path : str, optional
        path where the mirror configuration file is to be saved
    what : str, optional
        ["scope", "ao"] dump the mirror configuration for either the telescope
        or the AO module
    """

    if what.lower() == "scope":
        print("Dumping telescope mirror configuration.")
        fname = find_file("EC_mirrors_scope.tbl")
    elif what.lower() == "ao":
        print("Dumping AO mirror configuration.")
        fname = find_file("EC_mirrors_ao.tbl")

    if path is None:
        f = open(fname, "r")
        print(f.read())
        f.close()
    else:
        path = os.path.dirname(path)
        shutil.copy(fname, path)


def read_config(config_file):
    """
    Read in a SimMETIS configuration file

    The configuration file is in SExtractor format:
       'PARAMETER    Value    # Comment'

    Parameters
    ----------
    config_file : str
        the filename of the .config file

    Returns
    -------
    config_dict : dict (collections.OrderedDict)
        A dictionary with keys 'PARAMETER' and values 'Value'.

    Notes
    -----
    The values of the dictionary are strings and will have to be converted to
    the appropriate data type as they are needed.
    """

    import re

    # Read the file into a list of strings
    config_dict = OrderedDict()

    # remove lines that are all spaces or spaces + '#'
    # these are the regular expressions
    isempty = re.compile(r'^\s*$')
    iscomment = re.compile(r'^\s*#')

    with open(config_file, 'r') as fp1:
        for line in fp1:
            if isempty.match(line):
                continue
            if iscomment.match(line):
                continue

            line = line.rstrip()             # remove trailing \n
            content = line.split('#', 1)[0]  # remove comment
            param, value = content.split(None, 1)

            # Convert to number if possible
            try:
                config_dict[param] = float(value.strip())
            except ValueError:
                config_dict[param] = value.strip()

            # Convert string "none" to python None
            if isinstance(value, str) and value.lower() == "none":
                config_dict[param] = None

    return config_dict


def update_config(config_file, config_dict):
    """
    Update a SimMETIS configuration dictionary

    A configuration file in the SExtractor format:
    ::

      'PARAMETER    Value    # Comment'

    an existing configuration dictionary.

    Parameters
    ----------
    config_file : str
        the filename of the .config file

    Returns
    -------
    config_dict : dict
        A dictionary with keys 'PARAMETER' and values 'Value'.

    Notes
    -----
    the values of the dictionary are strings and will have
    to be converted to the appropriate data type as they are needed.

    """

    config_dict.update(read_config(config_file))

    return config_dict
