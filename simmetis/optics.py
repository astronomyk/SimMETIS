"""optics.py"""
###############################################################################
# OpticalTrain
#
# DESCRIPTION
# The OpticalTrain holds all the information regarding the optical path as
# well as the individual objects
#
# TODO List
# =========
# - Make the Detector independent of the OpticalTrain
# - Implement saving and reloading of OpticalTrain objects
#

import os

import glob
import warnings
import logging
#from datetime import datetime as dt    # unused
from copy import deepcopy

import numpy as np

from astropy.io import fits      # unused
from astropy.io import ascii as ioascii    # 'ascii' redefines built-in
import astropy.units as u

from . import psf as psf
from . import spectral as sc
from . import spatial as pe
from .source import flat_spectrum_sb, scale_spectrum_sb
from .commands import UserCommands
from .utils import __pkg_dir__

import pdb

__all__ = ["OpticalTrain", "get_filter_curve", "get_filter_set"]

class OpticalTrain(object):
    """
    The OpticalTrain object reads in or generates the information necessary to
    model the optical path for all (3) sources of photons: the astronomical
    source, the atmosphere and the primary mirror.

    Parameters
    ----------
    cmds : UserCommands, optional
        Holds the commands needed to generate a model of the optical train


    Optional Parameters
    -------------------
    Any keyword-value pair contained in the default configuration file


    See Also
    --------
    .commands.dump_defaults(), .commands.UserCommands


    General Attributes
    ------------------
    - cmds : commands, optional
        a dictionary of commands for running the simulation

    Spatial attributes (for PSFs)
    -----------------------------
    lam_bin_edges : 1D array
        [um] wavelengths of the edges of the bins used for each PSF layer
    lam_bin_centers
        [um] wavelengths of the centre of the bins used for each PSF layer
    pix_res : float
        [arcsec] infernal oversampled pixel resolution (NOT detector plate scale)
    psf :
    jitter_psf :
    adc_shifts
        [pixel]
    field_rot :
        [degrees]

    Spectral attributes (for EmissionCurves)
    ----------------------------------------
    lam : 1D array
        [um] Vector of wavelength bins for spectrals
    lam_res : float
        [um] resolution between
    psf_size : int
        [pixels] The width of a PSF
    tc_ao : TransmissionCurve
        [0..1]
    tc_mirror : TransmissionCurve
        [0..1]
    tc_atmo : TransmissionCurve
        [0..1]
    tc_source : TransmissionCurve
        [0..1]
    ec_ao : EmissionCurve
        [ph/s/voxel]
    ec_mirror : EmissionCurve
        [ph/s/voxel]
    ec_atmo : EmissionCurve
        [ph/s/voxel]
    ph_ao : EmissionCurve
        [ph/s/voxel]
    ph_mirror : EmissionCurve
        [ph/s/voxel]
    ph_atmo : EmissionCurve
        [ph/s/voxel]
    n_ph_ao : float
        [ph/s]
    n_ph_mirror : float
        [ph/s]
    n_ph_atmo : float
        [ph/s]


    """

    def __init__(self, cmds, **kwargs):
        self.info = dict([])

        if cmds is None:
            cmds = UserCommands()
        self.cmds = deepcopy(cmds)
        self.cmds.update(kwargs)

        self.tc_master = None   # set in separate method
        self.psf_size = None   # set in separate method

        fname = self.cmds["SIM_OPT_TRAIN_IN_PATH"]
        if fname is not None:
            if not os.path.exists(fname):
                raise ValueError(fname+" doesn't exist")

            self.read(fname)
        else:
            self.lam_bin_edges = cmds.lam_bin_edges
            self.lam_bin_centers = cmds.lam_bin_centers
            self.pix_res = cmds.pix_res

            self.lam = cmds.lam
            self.lam_res = cmds.lam_res

            self._make()


    def _make(self, cmds=None):
        """
        To make an optical system, cmds must contain all the keywords from the
        various .config files. 'cmds' should have been send to __init__, but any
        changes can be passed to make()

        Parameters
        ----------
        - cmds : commands
            a dictionary of commands
        """

        # Here we make the optical train. This includes
        # - the optical path for the source photons
        #   - master transmission curve             [can be exported]
        #       - atmosphere, n x mirror, instrument window, internal mirrors
        #       - dichroic, filter, detector QE
        #   - list of wave-dep plane effects
        #       - imperfect adc
        #   - master psf cube
        #       - AO psf - analytic or from file    [can be exported]
        #       - jitter psf                        [can be exported]
        #   - list of wave-indep plane effects
        #       - imperfect derotation
        #       - distortion                        [weight map can be exported]
        #       - flat fielding                     [can be exported]
        #   - detector
        #       - noise frame,

        if self.cmds.verbose:
            print("Generating an optical train")
        logging.debug("[OpticalTrain] Generating an optical train")

        if cmds is not None:
            self.cmds.update(cmds)
        self._load_all_tc()
        self._gen_all_tc()
        self.psf = self._gen_master_psf()

        # Get the ADC shifts, telescope shake and field rotation angle
        self.adc_shifts = self._gen_adc_shifts()
        self.jitter_psf = self._gen_telescope_shake()
        #self.field_rot = self._gen_field_rotation_angle()


    def replace_psf(self, new_psf, lam_bin_centers):
        """
        Change the PSF of the optical train
        """
        pass


    def update_filter(self, trans=None, lam=None, filter_name=None):
        """
        Update the filter curve without recreating the full OpticalTrain object

        Parameters
        ----------
        trans : TransmissionCurve, np.array, list, optional
            [0 .. 1] the transmission coefficients. Either a TransmissionCurve
            object can be passed (in which case omit ``lam``) or an array/list can
            be passed (in which case specify ``lam``)
        lam : np.array, list, optional
            [um] an array for the spectral bin centres, if ``trans`` is not a
            TransmissionCurve object
        filter_name : str, optional
            The name of a filter curve contained in the package_dir. User
            get_filter_set() to find which filter curves are installed.

        See also
        --------
        :class:`simcado.spectral.TransmissionCurve`
        :func:`simcado.optics.get_filter_set`

        """
        if filter_name == lam == trans == None:
            raise ValueError("At least one parameter must be specified")

        if filter_name is not None:
            filt = get_filter_curve(filter_name)
        elif trans is not None:
            if isinstance(trans, (sc.TransmissionCurve,
                                  sc.EmissionCurve,
                                  sc.UnityCurve,
                                  sc.BlackbodyCurve)):
                filt = trans
            elif isinstance(trans, (np.ndarray, list, tuple)) and \
                 isinstance(lam, (np.ndarray, list, tuple)):
                filt = sc.TransmissionCurve(lam=lam, val=trans,
                                            lam_res=self.lam_res)

        self.cmds["INST_FILTER_TC"] = filt
        self._gen_all_tc()


    def read(self, filename):
        pass

    def save(self, filename):
        pass


    def apply_tracking(self, arr):
        return pe.tracking(arr, self.cmds)

    def apply_derotator(self, arr):
        return pe.derotator(arr, self.cmds)

    def apply_wind_jitter(self, arr):
        return pe.wind_jitter(arr, self.cmds)

    def _load_all_tc(self, tc_list=None):
        """
        Pre-loads all the transmission curves
        """

        # Safe default - lists should not be defaults in declaration
        if tc_list is None:
            tc_list = ["ATMO_TC", "SCOPE_M1_TC", "INST_MIRROR_AO_TC",
                       "INST_ENTR_WINDOW_TC", "INST_DICHROIC_TC",
                       "INST_MIRROR_TC", "INST_ADC_TC",
                       "INST_PUPIL_TC", "INST_FILTER_TC", "FPA_QE"]

        for cur_tc in tc_list:
            if isinstance(self.cmds[cur_tc], str):
                airmass = self.cmds["ATMO_AIRMASS"] if cur_tc == "ATMO_TC" else None
                self.cmds[cur_tc] = sc.TransmissionCurve(filename=self.cmds[cur_tc],
                                                         airmass=airmass)
            elif self.cmds[cur_tc] is None:
                self.cmds[cur_tc] = sc.UnityCurve()

        # see Rics email from 22.11.2016
        wfe = self.cmds["INST_TOTAL_WFE"]
        lam = self.lam
        val = np.exp(-(2 * np.pi * (wfe*u.nm) / (lam*u.um))**2)
        self.cmds.cmds["INST_SURFACE_FACTOR"] = sc.TransmissionCurve(lam=lam,
                                                                     val=val)

    def _gen_thermal_emission(self):
        '''Number of thermal photons emitted by the warm surfaces

        Returns
        -------

        A tuple with the total number of photons in the wavelength range
        and the spectrum.
        '''

        # List of warm mirrors
        mirr_list = self.cmds.mirrors_telescope

        # Load transmission curves into a dictionary indexed by coating
        tc_dict = dict()
        for coating in np.unique(mirr_list['Coating']):
            if os.path.exists(coating):
                tc_file = coating
            elif os.path.exists(os.path.join(__pkg_dir__, "data", coating)):
                tc_file = os.path.join(__pkg_dir__, "data", coating)
            else:
                raise ValueError("Could not find file: "+coating)

            tc_dict[coating] = sc.TransmissionCurve(tc_file)

        # Follow the thermal flux through all the warm elements
        total_flux = 0
        if self.cmds.verbose:
            print("Total flux init: " + str(total_flux))

        # The etendue is assumed to be conserved throughout the system.
        # We used to compute the solid angle at each mirror, given its area.
        mirror = mirr_list[0]
        etendue = (mirror['Outer']**2 - mirror['Inner']**2) * np.pi/4 \
                  * self.pix_res**2

        for mirror in mirr_list:
            # mirror area projected perpendicular to beam
            area = (mirror['Outer']**2 - mirror['Inner']**2) * np.pi/4 \
                   * np.cos(np.deg2rad(mirror['Angle']))
            angle = np.sqrt(etendue / area)
            temp = mirror['Temp']
            reflectivity = tc_dict[mirror['Coating']].val
            emissivity = 1. - reflectivity

            mirror_flux = sc.BlackbodyCurve(lam=self.lam, temp=temp,
                                            pix_res=angle, area=area) * \
                          emissivity

            total_flux = mirror_flux + total_flux * reflectivity
            n_ph_thermal = total_flux.photons_in_range(self.lam_bin_edges[0],
                                                       self.lam_bin_edges[-1])

            if self.cmds.verbose:
                print("{0}: {1}   Mean reflectivity {2:.3f}".format(
                    mirror['Mirror'], mirror['Coating'],
                    np.mean(reflectivity)))
                print("{0}: Emitted {1:.3f}     Total {2:.3f}".format(
                    mirror['Mirror'],
                    mirror_flux.photons_in_range(self.lam_bin_edges[0],
                                                 self.lam_bin_edges[-1]),
                    total_flux.photons_in_range(self.lam_bin_edges[0],
                                                self.lam_bin_edges[-1])))

        return n_ph_thermal, total_flux


    def _gen_all_tc(self):

        ############## AO INSTRUMENT PHOTONS #########################
        if self.cmds.verbose:
            print("Generating AO module mirror emission photons")
        logging.debug("[_gen_all_tc] Generating AO module mirror emission photons")

        # get the total area of mirrors in the telescope
        # !!!!!! Bad practice, this is E-ELT specific hard-coding !!!!!!

        mirr_list = self.cmds.mirrors_ao
        ao_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                     mirr_list["Inner"]**2)

        # Make the transmission curve for the blackbody photons from the mirror
        self.tc_ao = self._gen_master_tc(preset="ao")
        self.ec_ao = sc.BlackbodyCurve(lam    =self.tc_ao.lam,
                                       temp   =self.cmds["INST_AO_TEMPERATURE"],
                                       pix_res=self.cmds.pix_res,
                                       area   =ao_area)
        # Really dodgy hack to emulate emissivity - half way between Al and AgAl
        self.ec_ao *= 0.1

        if self.cmds["INST_USE_AO_MIRROR_BG"].lower() == "yes" and \
           self.cmds["SCOPE_PSF_FILE"].lower() != "scao":

            self.ph_ao = self.ec_ao * self.tc_ao
            self.n_ph_ao = self.ph_ao.photons_in_range(self.lam_bin_edges[0],
                                                       self.lam_bin_edges[-1])
        else:
            self.ec_ao = None
            self.ph_ao = None
            self.n_ph_ao = 0.



        ############## TELESCOPE PHOTONS #########################
        if self.cmds.verbose:
            print("Generating telescope mirror emission photons")
        logging.debug("[_gen_all_tc] Generating telescope mirror emission photons")

        # get the total area of mirrors in the telescope
        # !!!!!! Bad practice, this is E-ELT specific hard-coding !!!!!!
        mirr_list = self.cmds.mirrors_telescope
        scope_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                        mirr_list["Inner"]**2)


        if "Temp" in mirr_list.colnames:
        	##
        	## KL/LB 25 June: manually adding SCOPE_TEMP key to user commands object
        	##                since we have taken it out of the config files
            self.cmds.cmds["SCOPE_TEMP"] = mirr_list["Temp"][0]
        # Make the transmission curve for the blackbody photons from the mirror
        self.tc_mirror = self._gen_master_tc(preset="mirror")
        self.ec_mirror = sc.BlackbodyCurve(lam    =self.tc_mirror.lam,
                                           temp   =self.cmds["SCOPE_TEMP"],
                                           pix_res=self.cmds.pix_res,
                                           area   =scope_area)

        if self.cmds["SCOPE_USE_MIRROR_BG"].lower() == "yes":
            # KL - _gen_thermal_emission() returns the sum of all thermal photons
            # not just the ones that pass through the system transmission curve
            # Add the 3rd line here to correct this
            self.n_ph_mirror, self.ec_mirror = self._gen_thermal_emission()
            self.ph_mirror   = self.ec_mirror * self.tc_mirror
            self.n_ph_mirror = self.ph_mirror.photons_in_range(self.lam_bin_edges[0],
                                                               self.lam_bin_edges[-1])
        else:
            self.ec_mirror = None
            self.ph_mirror = None
            self.n_ph_mirror = 0.


        ############## ATMOSPHERIC PHOTONS #########################
        if self.cmds.verbose:
            print("Generating atmospheric emission photons")
        logging.debug("[_gen_all_tc] Generating atmospheric emission photons")

        # Make the spectral curves for the atmospheric background photons
        self.tc_atmo = self._gen_master_tc(preset="atmosphere")

        if self.cmds["ATMO_USE_ATMO_BG"].lower() == "yes":
            if self.cmds["ATMO_EC"] is not None:

                # self.ec_atmo = sc.EmissionCurve(filename=self.cmds["ATMO_EC"],
                                                # pix_res=self.cmds.pix_res,
                                                # area=self.cmds.area,
                                                # airmass=self.cmds["ATMO_AIRMASS"])

                # self.th_atmo = sc.BlackbodyCurve(lam    =self.ec_atmo.lam,
                                                 # temp   =self.cmds["ATMO_TEMPERATURE"],
                                                 # pix_res=self.cmds.pix_res,
                                                 # area   =scope_area)

                # self.ec_atmo += self.th_atmo

                # print("Just loaded EC")
                # print((self.tc_atmo * self.ec_atmo).photons_in_range()

                ################################################################
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ################################################################
                # SUPER DODGY HACK THAT NEEDS TO BY TRACKED DOWN AT SOME POINT!!
                # But it brings SImCADO inline with the HAWKI ETC and SkyCalc
                # It's only the Sky BG emission that is affected.
                # The mirror emission is fine
                ################################################################
                # self.ec_atmo *= 2.5
                ################################################################


                self.ec_atmo = sc.get_sky_spectrum(fname=self.cmds["ATMO_EC"],
                                               airmass=self.cmds["ATMO_AIRMASS"],
                                               return_type="emission",
                                               area=self.cmds.area,
                                               pix_res=self.cmds.pix_res)
                lam = self.ec_atmo.lam
                val = self.ec_atmo.val

                sky_mag = self.cmds["ATMO_BG_MAGNITUDE"]
                if sky_mag is not None and isinstance(sky_mag, (float, int)):
                    lam, val = scale_spectrum_sb(lam=lam, spec=val,
                                                 filter_name=self.cmds["INST_FILTER_TC"],
                                                 mag_per_arcsec=sky_mag,
                                                 pix_res=self.cmds.pix_res,
                                                 return_ec=False)

                    self.ec_atmo = sc.EmissionCurve(lam=lam, val=val,
                                                 pix_res=self.cmds.pix_res,
                                                 area=self.cmds.area,
                                                 units="ph/(s m2)",
                                                 airmass=self.cmds["ATMO_AIRMASS"])
            else:
                ################## TODO ######################
                # Generalise this to accept any TransmissionCurve object
                from .source import flat_spectrum_sb
                self.ec_atmo = flat_spectrum_sb(self.cmds["ATMO_BG_MAGNITUDE"],
                                                self.cmds["INST_FILTER_TC"],
                                                self.cmds["SIM_DETECTOR_PIX_SCALE"],
                                                return_ec=True)
                self.ec_atmo *= self.cmds.area

            self.ph_atmo = self.tc_atmo * self.ec_atmo
            self.n_ph_atmo = self.ph_atmo.photons_in_range(self.lam_bin_edges[0],
                                                           self.lam_bin_edges[-1])

        else:
            self.ec_atmo = None
            self.ph_atmo = None
            self.n_ph_atmo = 0.

        self.n_ph_bg = self.n_ph_atmo + self.n_ph_mirror + self.n_ph_ao

        ############## SOURCE PHOTONS #########################
        if self.cmds.verbose:
            print("Generating optical path for source photons")
        logging.debug("[_gen_all_tc] GGenerating optical path for source photons")

        # Make the transmission curve and PSF for the source photons
        self.tc_source = self._gen_master_tc(preset="source")


    def _gen_master_tc(self, tc_keywords=None, preset=None):
        """
        Combine a list of TransmissionCurves into one, either by specifying the
        list of command keywords (e.g. ATMO_TC) or by passing a preset keywords

        Optional Parameters
        ===================
        tc_keywords: a list of keywords from the .config files. E.g:
                     tc_keywords = ['ATMO_TC', 'SCOPE_M1_TC', 'INST_FILTER_TC']
        preset: a present string for the most common collections of keywords:
                - 'source' includes all the elements seen by source photons
                - 'atmosphere' includes surfaces seen by the atmospheric BG
                - 'mirror' includes surfaces seen by the M1 blackbody photons
                - 'ao' inludes surfaces seen by the AO module blackbody photons
        """

        if tc_keywords is None:
            if preset is not None:
                base = ['INST_MIRROR_AO_TC'] * (int(self.cmds['INST_NUM_AO_MIRRORS']) - 1) + \
                       ['INST_ENTR_WINDOW_TC'] * int(self.cmds['INST_ENTR_NUM_SURFACES']) + \
                       ['INST_DICHROIC_TC']  * int(self.cmds['INST_DICHROIC_NUM_SURFACES']) + \
                       ['INST_MIRROR_TC']    * (int(self.cmds['INST_NUM_MIRRORS'])) + \
                       ['INST_ADC_TC']       * int(self.cmds['INST_ADC_NUM_SURFACES']) + \
                       ['INST_PUPIL_TC']     * int(self.cmds['INST_PUPIL_NUM_SURFACES']) + \
                       ['INST_FILTER_TC'] + \
                       ['INST_SURFACE_FACTOR'] + \
                       ['FPA_QE']

                ao = ['INST_MIRROR_AO_TC'] + ['SCOPE_M1_TC'] * (len(self.cmds.mirrors_telescope) - 1)

                if preset == "ao":
                    tc_keywords = base
                elif preset == "mirror":
                    tc_keywords = base + ao
                elif preset == "atmosphere":
                    tc_keywords = base + ao + ['SCOPE_M1_TC']
                elif preset == "source":
                    tc_keywords = base + ao + ['SCOPE_M1_TC', 'ATMO_TC']
                else:
                    raise ValueError("Unknown preset parameter " + preset)

            else:
                warnings.warn("""
                No presets or keywords passed to gen_master_tc().
                Setting self.tc_master = sc.UnityCurve()""", UserWarning)
                self.tc_master = sc.UnityCurve()
                return

        tc_dict = dict([])

        for key in tc_keywords:
            if key not in self.cmds.keys():
                raise ValueError(key + " is not in your list of commands")

            if self.cmds[key] is not None:
                if isinstance(self.cmds[key], (sc.TransmissionCurve,
                                               sc.EmissionCurve,
                                               sc.UnityCurve,
                                               sc.BlackbodyCurve)):
                    tc_dict[key] = self.cmds[key]
                else:
                    airmass = self.cmds["ATMO_AIRMASS"] if key == "ATMO_TC" else None
                    tc_dict[key] = sc.TransmissionCurve(filename=self.cmds[key],
                                                        lam_res=self.lam_res,
                                                        airmass=airmass)
            else:
                tc_dict[key] = sc.UnityCurve()

        tc_master = sc.UnityCurve(lam=self.lam, lam_res=self.lam_res,
                                  min_step=self.cmds["SIM_SPEC_MIN_STEP"])
        for key in tc_keywords:
            tc_master *= tc_dict[key]

        self.tc_keywords = tc_keywords
        self.tc_dict = tc_dict

        return tc_master


    def _gen_master_psf(self):
        """
        Import or make a master PSF for the system.

        Notes
        -----
        Jitter can be applied to detector array as a single PSF, and the
               ADC shift can be applied to each layer of the psf separately

        """

        ############################################################
        # !!!!!!!!! USER DEFINED CUBE NOT FULLY TESTED !!!!!!!!!!! #
        # !!!!!!! Analytic still using Airy (x) Gaussian !!!!!!!!! #
        # !!!!!!!!!!!!!!! Implement Moffat PSF !!!!!!!!!!!!!!!!!!! #
        ############################################################

        self.psf_size = self.cmds["SIM_PSF_SIZE"]

        # Make a PSF for the main mirror. If there is one on file, read it in
        # otherwise generate an Airy+Gaussian (or Moffat, Oliver?)

        if self.cmds["SCOPE_PSF_FILE"] is None:
            warnings.warn("""
            SCOPE_PSF_FILE == None.
            Generating Moffat profile from with FWHM = OBS_SEEING""")
            logging.debug("No PSF Given: making Seeing PSF")

            hdulist = fits.HDUList()
            for lam in self.cmds.lam_bin_centers:

                psf_mo = psf.seeing_psf(fwhm   =self.cmds["OBS_SEEING"],
                                        size   =self.cmds["SIM_PSF_SIZE"],
                                        pix_res=self.cmds["SIM_DETECTOR_PIX_SCALE"],
                                        psf_type="moffat", filename=None)

                psf_mo[0].header["WAVELENG"] = lam
                hdulist.append(psf_mo[0])

            psf_m1 = psf.UserPSFCube(hdulist, self.lam_bin_centers)


        elif isinstance(self.cmds["SCOPE_PSF_FILE"], psf.PSFCube):
            psf_m1 = self.cmds["SCOPE_PSF_FILE"]
            #logging.debug("Using PSF: " + self.cmds["SCOPE_PSF_FILE"])

        elif isinstance(self.cmds["SCOPE_PSF_FILE"], str):
            if self.cmds.verbose:
                print("Using PSF:", self.cmds["SCOPE_PSF_FILE"])

            if os.path.exists(self.cmds["SCOPE_PSF_FILE"]):
                #logging.debug("Using PSF: " + self.cmds["SCOPE_PSF_FILE"])

                psf_m1 = psf.UserPSFCube(self.cmds["SCOPE_PSF_FILE"],
                                         self.lam_bin_centers)

                if psf_m1[0].pix_res != self.pix_res:
                    psf_m1.resample(self.pix_res)
            else:
                warnings.warn("""
                Couldn't resolve SCOPE_PSF_FILE.
                Returning an Delta function for SCOPE_PSF_FILE""")

                psf_m1 = psf.DeltaPSFCube(self.lam_bin_centers,
                                          pix_res=self.pix_res,
                                          size=9)
                logging.debug("Couldn't resolve given PSF: making Delta PSF")

        return psf_m1


    def _gen_adc_shifts(self):
        """
        Keywords:
        """
        adc_shifts = pe.adc_shift(self.cmds)
        return adc_shifts


    def _gen_field_rotation_angle(self):
        return 0


    def _gen_telescope_shake(self):
        """
        Keywords:
        """
        jitter_psf = psf.GaussianPSF(fwhm=self.cmds["SCOPE_JITTER_FWHM"],
                                     pix_res=self.cmds.pix_res)
        return jitter_psf


## note: 'filter' redefines a built-in and should not be used
def get_filter_curve(filter_name):
    """
    Return a Vis/NIR broadband filter TransmissionCurve object

    Parameters
    ----------
    filter_name : str

    Notes
    -----
    Acceptable filters can be found be calling get_filter_set()
    """

    if filter_name not in get_filter_set(path=None):
        raise ValueError("filter not recognised: "+filter)
    fname = os.path.join(__pkg_dir__, "data", "TC_filter_"+filter_name+".dat")
    return sc.TransmissionCurve(filename=fname)


def get_filter_set(path=None):
    """
    Return a list of the filters installed in the package directory
    """
    if path is None:
        path = os.path.join(__pkg_dir__, "data")
    lst = [i.replace(".dat", "").split("TC_filter_")[-1] \
                    for i in glob.glob(os.path.join(path, "TC_filter*.dat"))]
    return lst
