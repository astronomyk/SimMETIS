# pylint: disable=too-many-lines
"""
The module that contains the functionality to create Source objects

Summary
-------
The source is essentially a list of spectra and a list of positions. The
list of positions contains a reference to the relevant spectra. The advantage
here is that if there are repeated spectra in a data cube, we can reduce the
amount of calculations needed. Furthermore, if the input is originally a list
of stars, etc., where the position of a star is not always an integer multiple
of the plate scale, we can keep the information until the PSFs are needed.

Classes
-------
Source

Functions
---------
Functions to create ``Source`` objects
::

    empty_sky()
    star(mag, filter_name="Ks", spec_type="A0V", x=0, y=0)
    stars(mags, filter_name="Ks", spec_types=["A0V"], x=[0], y=[0])
    star_grid(n, mag_min, mag_max, filter_name="Ks", separation=1, area=1,
              spec_type="A0V")
    source_from_image(images, lam, spectra, pix_res, oversample=1,
                      units="ph/s/m2", flux_threshold=0,
                      center_pixel_offset=(0, 0))
    source_1E4_Msun_cluster(distance=50000, half_light_radius=1)


Functions for manipulating spectra for a ``Source`` object
::

    scale_spectrum(lam, spec, mag, filter_name="Ks", return_ec=False)
    scale_spectrum_sb(lam, spec, mag_per_arcsec, pix_res=0.004,
                      filter_name="Ks", return_ec=False)
    flat_spectrum(mag, filter_name="Ks", return_ec=False)
    flat_spectrum_sb(mag_per_arcsec, filter_name="Ks", pix_res=0.004,
                     return_ec=False)


Functions regarding photon flux and magnitudes
::

    zero_magnitude_photon_flux(filter_name)
    _get_stellar_properties(spec_type, cat=None, verbose=False)
    _get_stellar_mass(spec_type)
    _get_stellar_Mv(spec_type)
    _get_pickles_curve(spec_type, cat=None, verbose=False)


Helper functions
::

    value_at_lambda(lam_i, lam, val, return_index=False)
    SED(spec_type, filter_name="V", magnitude=0.)

"""
###############################################################################
# source
#
# DESCRIPTION

#
# The source contains two arrays:
#  - PositionArray:
#  - SpectrumArray


# Flow of events
# - Generate the lists of spectra and positions
# - Apply the transmission curves [SpectrumArray]
# - shrink the 1D spectra to the resolution of the psf layers [SpectrumArray]
# - apply any 2D spatials [PositionArray]
# for i in len(slices)
#   - generate a working slice [PositionArray, SpectrumArray, WorkingSlice]
#   - apply the PSF for the appropriate wavelength [WorkingSlice]
#   - apply any wavelength dependent spatials [WorkingSlice]
#   - apply Poisson noise to the photons in the slice [WorkingSlice]
#   - add the WorkingSlice to the FPA [WorkingSlice, FPArray]


## TODO implement conversions to Source object from:
# ascii
#    x, y, mag, [temp]
#    x, y, type
# images
#    JHK
#    cube


import os

import warnings
from copy import deepcopy
from glob import glob

import numpy as np
import scipy.ndimage.interpolation as spi
from scipy.signal import fftconvolve

from astropy.io import fits
from astropy.io import ascii as ioascii
from astropy.convolution import convolve
import astropy.units as u
import astropy.constants as c

from .spectral import TransmissionCurve, EmissionCurve, UnityCurve, BlackbodyCurve
from . import psf as sim_psf
from . import utils
from .utils import __pkg_dir__

__all__ = ["Source",
           "star", "stars", "cluster",
           "spiral", "spiral_profile", "elliptical", "sersic_profile"
           "source_from_image",
           "star_grid", "empty_sky", "SED",
           "sie_grad", "apply_grav_lens"
           "get_SED_names",
           "scale_spectrum", "scale_spectrum_sb",
           "flat_spectrum", "flat_spectrum_sb",
           "value_at_lambda",
           "BV_to_spec_type",
           "zero_magnitude_photon_flux", "mag_to_photons", "photons_to_mag",
           "_get_pickles_curve", "_get_stellar_properties",
           "_get_stellar_Mv", "_get_stellar_mass"]


# add_uniform_background() moved to detector
# get_slice_photons() renamed to photons_in_range() and moved to Source
# _apply_transmission_curve() moved to Source
# apply_optical_train() moved to Source


class Source(object):
    """
    Create a source object from a file or from arrays

    Source class generates the arrays needed for source. It takes various
    inputs and converts them to an array of positions and references to spectra.
    It also converts spectra to photons/s/voxel. The default units for input
    data is ph/s/m2/bin.

    The internal variables are related like so:
    ::
        f(x[i], y[i]) = spectra[ref[i]] * weight[i]


    Parameters
    ----------
    filename : str
        FITS file that contains either a previously saved ``Source`` object or a
        data cube with dimensions x, y, lambda. A ``Source`` object is
        identified by the header keyword SIMMETIS with value SOURCE.

    or

    lam : np.array
        [um] Wavelength bins of length (m)
    spectra : np.array
        [ph/s/m2/bin] A (n, m) array with n spectra, each with m spectral bins
    x, y : np.array
        [arcsec] coordinates of where the emitting sources are relative to the
        centre of the field of view
    ref : np.array
        the index for .spectra which connects a position (x, y) to a spectrum
        f(x[i], y[i]) = spectra[ref[i]] * weight[i]
    weight : np.array
        A weighting to scale the relevant spectrum for each position

    Keyword arguments
    -----------------
    units : str
        The units of the spectra. Default is ph/s/m2/bin
    pix_unit : str
        Default is "arcsec". Acceptable are "arcsec", "arcmin", "deg", "pixel"
    exptime : float
        If the input spectrum is not normalised to 1 sec
    area : float
        The telescope area used to generate the source object
    pix_res : float
        [arcsec] The pixel resolution of the detector. Useful for surface
        brightness calculations
    bg_spectrum : EmissionCurve
        If there is a surface brightness term to add, add it here

    """

    def __init__(self, filename=None,
                 lam=None, spectra=None, x=None, y=None, ref=None, weight=None,
                 **kwargs):

        self.params = {"units"   : "ph/s",
                       "pix_unit": "arcsec",
                       "exptime" : 1,
                       "area"    : 1,
                       "pix_res" : 0.004,
                       "bg_spectrum" : None}
        self.params.update(kwargs)

        if isinstance(x, (tuple, list)):
            x = np.array(x)
        if isinstance(y, (tuple, list)):
            y = np.array(y)

        if x is not None:
            x = x.astype(np.float32)
        if y is not None:
            y = y.astype(np.float32)

        if "pix" in self.params["pix_unit"]:
            x *= self.params["pix_res"]
            y *= self.params["pix_res"]
        elif "arcmin" in self.params["pix_unit"]:
            x *= 60.
            y *= 60.
        elif "deg" in self.params["pix_unit"]:
            x *= 3600.
            y *= 3600.


        self.info = dict([])
        self.info['description'] = "List of spectra and their positions"

        self.units = u.Unit(self.params["units"])
        self.exptime = self.params["exptime"]
        self.pix_res = self.params["pix_res"]

        self.x = None
        self.y = None  # set later

        # A file can contain a previously saved Source object; in this case the header keyword
        # "SIMMETIS" is set to "SOURCE". If this is not the case, we assume that the file
        # contains a data cube with dimensions x, y, lambda.
        # If no filename is given, we build the Source from the arrays.
        if filename is not None:
            hdr = fits.getheader(filename)
            if "SIMMETIS" in hdr.keys() and hdr["SIMMETIS"] == "SOURCE":
                self.read(filename)
            else:
                self._from_cube(filename)
        elif not any(elem is None for elem in (lam, spectra, x, y, ref)):
            self._from_arrays(lam, spectra, x, y, ref, weight)
        else:
            raise ValueError("Trouble with inputs. Could not create Source")

        self.ref = np.array(self.ref, dtype=int)
        self.x_orig = deepcopy(self.x)
        self.y_orig = deepcopy(self.y)
        self.spectra_orig = deepcopy(self.spectra)

        self.bg_spectrum = None


    @classmethod
    def load(cls, filename):
        '''Load :class:'Source' object from filename'''
        import pickle
        with open(filename, 'rb') as fp1:
            src = pickle.load(fp1)
        return src


    def dump(self, filename):
        '''Save to filename as a pickle'''
        import pickle
        with open(filename, 'wb') as fp1:
            pickle.dump(self, fp1)


    def apply_optical_train(self, opt_train, detector, chips="all",
                            sub_pixel=False, **kwargs):
        """
        Apply all effects along the optical path to the source photons

        Parameters
        ----------
        opt_train : simmetis.OpticalTrain
            the object containing all information on what is to happen to the
            photons as they travel from the source to the detector
        detector : simmetis.Detector
            the object representing the detector
        chips : int, str, list, optional
            The IDs of the chips to be readout. "all" is also acceptable
        sub_pixel : bool, optional
            if sub-pixel accuracy is needed, each source is shifted individually.
            Default is False

        Other Parameters
        ----------------
        INST_DEROT_PERFORMANCE : float
            [0 .. 100] Percentage of the sky rotation that the derotator removes
        SCOPE_JITTER_FWHM : float
            [arcsec] The FWMH of the gaussian blur caused by jitter
        SCOPE_DRIFT_DISTANCE : float
            [arcsec] How far from the centre of the field of view has the
            telescope drifted during a DIT

        Notes
        -----
        Output array is in units of [ph/s/pixel] where the pixel is internal
        oversampled pixels - not the pixel size of the detector chips

        """
        params = {"verbose"                : opt_train.cmds.verbose,
                  "INST_DEROT_PERFORMANCE" : opt_train.cmds["INST_DEROT_PERFORMANCE"],
                  "SCOPE_JITTER_FWHM"      : opt_train.cmds["SCOPE_JITTER_FWHM"],
                  "SCOPE_DRIFT_DISTANCE"   : opt_train.cmds["SCOPE_DRIFT_DISTANCE"],
                  "sub_pixel"              : sub_pixel}
        params.update(self.params)
        params.update(kwargs)

        self.pix_res = opt_train.pix_res

        # 1. Apply the master transmission curve to all the spectra
        #
        # 1.5 Create a canvas onto which we splat the PSFed sources
        #
        # 2. For each layer between cmds.lam_bin_edges[i, i+1]
        #   - Apply the x,y shift for the ADC
        #       - Apply any other shifts
        #   - Apply the PSF for the layer
        #       - Sum up the photons in this section of the spectra
        #   - Add the layer to the final image array
        #
        # 3. Apply wave-indep psfs
        #   - field rotation
        #   - telescope shake
        #   - tracking error
        #
        # 3.5 Up until now everything is ph/s/m2/bin
        #     Apply the collecting area of the telescope
        #
        # 4. Add the average number of atmo-bg and mirror-bb photons
        # 5. Apply the instrumental distortion

        if chips is None or str(chips).lower() == "all":
            chips = np.arange(len(detector.chips))

        if not hasattr(chips, "__len__"):
            chips = [chips]

        # 1.
        self._apply_transmission_curve(opt_train.tc_source)

        for chip_i in chips:
            print("Generating image for chip", detector.chips[chip_i].id)

            # 1.5
            image = None

            # 2.
            for i in range(len(opt_train.lam_bin_edges[:-1])):

                if params["verbose"]:
                    print("Wavelength slice [um]:",
                          opt_train.lam_bin_centers[i])

                # apply the adc shifts
                self._x = self.x + opt_train.adc_shifts[0][i]
                self._y = self.y + opt_train.adc_shifts[1][i]

                # include any other shifts here


                # apply the psf (get_slice_photons is called within)
                lam_min, lam_max = opt_train.lam_bin_edges[i:i+2]
                psf_i = utils.nearest(opt_train.psf.lam_bin_centers,
                                      opt_train.lam_bin_centers[i])
                psf = opt_train.psf[psf_i]


                oversample = opt_train.cmds["SIM_OVERSAMPLING"]
                sub_pixel = params["sub_pixel"]
                verbose = params["verbose"]

                # image is in units of ph/s/pixel/m2
                imgslice = self.image_in_range(psf, lam_min, lam_max,
                                               detector.chips[chip_i],
                                               pix_res=opt_train.pix_res,
                                               oversample=oversample,
                                               sub_pixel=sub_pixel,
                                               verbose=verbose)
                if image is None:
                    image = imgslice
                else:
                    image += imgslice

            # 3. Apply wavelength-independent spatial effects
            # !!!!!!!!!!!!!! All of these need to be combined into a single
            # function that traces out the path taken by the telescope,
            # rather than having the arcs from the derotator() function
            # being stretched by the tracking() function and then the whole
            # thing blurred by wind_jitter()
            if params["INST_DEROT_PERFORMANCE"] < 100:
                image = opt_train.apply_derotator(image)
            if params["SCOPE_DRIFT_DISTANCE"] > 0.33 * self.pix_res:
                image = opt_train.apply_tracking(image)
            if params["SCOPE_JITTER_FWHM"] > 0.33 * self.pix_res:
                image = opt_train.apply_wind_jitter(image)

            # 3.5 Scale by telescope area
            image *= opt_train.cmds.area

            # 4. Add backgrounds
            image += (opt_train.n_ph_atmo + opt_train.n_ph_mirror +
                      opt_train.n_ph_ao)

            ## TODO: protected members should not be set by another class (OC)
            ##       These could be added to info dictionary, if they're only
            ##       informational.
            detector._n_ph_atmo = opt_train.n_ph_atmo
            detector._n_ph_mirror = opt_train.n_ph_mirror
            detector._n_ph_ao = opt_train.n_ph_ao

            # 5. Project onto chip
            self.project_onto_chip(image, detector.chips[chip_i])

        ######################################
        # CAUTION WITH THE PSF NORMALISATION #
        ######################################


    def project_onto_chip(self, image, chip):
        """
        Re-project the photons onto the same grid as the detectors use

        Parameters
        ----------
        image : np.ndarray
            the image to be re-projected
        chip : detector.Chip
            the chip object where the image will land
        """
        # This is just a change of pixel scale
        chip.reset()
        scale_factor = self.pix_res / chip.pix_res
        chip_arr = spi.zoom(image, scale_factor, order=1)
        chip_arr *= np.sum(image) / np.sum(chip_arr)
        chip.add_signal(chip_arr)


    def image_in_range(self, psf, lam_min, lam_max, chip, **kwargs):
        """
        Find the sources that fall in the chip area and generate an image for
        the wavelength range [lam_min, lam_max)

        Output is in [ph/s/pixel]

        Parameters
        ----------
        psf : psf.PSF object
            The PSF that the sources will be convolved with
        lam_min, lam_max : float
            [um] the wavelength range relevant for the psf
        chip : str, detector.Chip
            - detector.Chip : the chip that will be seeing this image.
            - str : ["tiny", "small", "center"] -> [128, 1024, 4096] pixel chips


        Optional parameters (**kwargs)
        ------------------------------
        sub_pixel : bool
            if sub-pixel accuracy is needed, each source is shifted individually.
            Default is False
        pix_res : float
            [arcsec] the field of view of each pixel. Default is 0.004 arcsec
        oversample : int
            the psf images will be oversampled to better conserve flux.
            Default is 1 (i.e. not oversampled)
        verbose : bool
            Default that of the OpticalTrain object

        Returns
        -------
        slice_array : np.ndarray
            the image of the source convolved with the PSF for the given range

        """

        params = {"pix_res"     :0.004,
                  "sub_pixel"   :False,
                  "oversample"  :1,
                  "verbose"     :False}

        params.update(kwargs)

        #  no PSF given: use a delta kernel
        if isinstance(psf, type(None)):
            psf = np.zeros((7, 7))
            psf[3, 3] = 1

        # psf cube given: extract layer for central wavelength
        if isinstance(psf, (sim_psf.PSFCube, sim_psf.UserPSFCube)):
            lam_cen = (lam_max + lam_min) / 2.
            psf = psf.nearest(lam_cen)

        # psf given as array: convert to PSF object
        if isinstance(psf, np.ndarray):
            arr = deepcopy(psf)
            pix_res = params["pix_res"] / params["oversample"]
            size = psf.shape[0]
            psf = sim_psf.PSF(size, pix_res)
            psf.set_array(arr)


        # TODO: There is no provision for chip rotation wrt (x, y) system (OC)
        # Create Chip object if chip described by a string
        if isinstance(chip, str):
            if chip.lower() == "small":
                from .detector import Chip
                chip = Chip(0, 0, 1024, 1024, 0.004)
            elif "cent" in chip.lower():
                from .detector import Chip
                chip = Chip(0, 0, 4096, 4096, 0.004)
            elif "tiny" in chip.lower():
                from .detector import Chip
                chip = Chip(0, 0, 128, 128, 0.004)
            else:
                raise ValueError("Unknown chip identification")

        # Check whether _x has been created - _x contains the adc corrections
        if not hasattr(self, "_x"):
            self._x = np.copy(self.x)
            self._y = np.copy(self.y)

        # Determine x- and y- range covered by chip
        # TODO: Use chip.wcs to convert (x, y) into pixel coordinates,
        #       then simply cut at the pixel edges. Alternatively,
        #       project chip edges to the sky.
        if chip is not None:
            mask = (self._x > chip.x_min) * (self._x < chip.x_max) * \
                   (self._y > chip.y_min) * (self._y < chip.y_max)
            params["pix_res"] = chip.pix_res / params["oversample"]
            x_min, x_max = chip.x_min, chip.x_max,
            y_min, y_max = chip.y_min, chip.y_max
            x_cen, y_cen = chip.x_cen, chip.y_cen

            naxis1, naxis2 = chip.naxis1, chip.naxis2

        else:
            # no chip given: use area covered by object arrays
            mask = np.array([True] * len(self._x))
            params["pix_res"] /= params["oversample"]
            x_min, x_max = np.min(self._x), np.max(self._x)
            y_min, y_max = np.min(self._y), np.max(self._y),
            x_cen, y_cen = (x_max + x_min) / 2, (y_max + y_min) / 2

            # the conversion to int was causing problems because some
            # values were coming out at 4095.9999, so the array was (4095, 4096)
            # hence the 1E-3 on the end
            naxis1 = int((x_max - x_min) / params["pix_res"] + 1E-3)
            naxis2 = int((y_max - y_min) / params["pix_res"] + 1E-3)

        slice_array = np.zeros((naxis1, naxis2), dtype=np.float32)
        slice_photons = self.photons_in_range(lam_min, lam_max)

        # convert point source coordinates to pixels
        x_pix = (self._x - x_cen) / params["pix_res"]
        y_pix = (self._y - y_cen) / params["pix_res"]

        self.x_pix = x_pix + chip.naxis1 // 2
        self.y_pix = y_pix + chip.naxis2 // 2

        # if sub-pixel accuracy is needed, be prepared to wait. For this we
        # need to go through every source spectrum in turn, shift the psf by
        # the decimal amount given by pos - int(pos), then place a
        # certain slice of the psf on the output array.
        ax, ay = np.array(slice_array.shape) // 2
        bx, by = np.array(psf.array.shape)   // 2
        mx, my = np.array(psf.array.shape) % 2

        if params["verbose"]:
            print("Chip ID:", chip.id,
                  "- Creating layer between [um]:", lam_min, lam_max)

        psf_array = np.copy(psf.array)

        if params["sub_pixel"] is True:
            # for each point source in the list, add a psf to the slice_array
            #x_int, y_int = np.floor(x_pix), np.floor(y_pix)
            #dx, dy = src.x - x_int, src.y - y_int

            if bx == ax and by == ay:
                pass
            elif bx > ax and by > ay:
                # psf_array larger than slice_array: cut down
                psf_array = psf_array[(bx - ax):(bx + ax), (by - ay):(by + ay)]
            elif bx < ax and by < ay:
                # psf_array smaller than slice_array: pad with zeros
                pad_x, pad_y = ax - bx, ay - by
                psf_array = np.pad(psf_array,
                                   ((pad_x, pad_x-mx),
                                    (pad_y, pad_y-my)),
                                   mode="constant")
            else:
                print("PSF", psf.array.shape, "Chip", slice_array.shape)
                raise ValueError("PSF and Detector chip sizes are odd:")

            for i in range(len(x_pix)):
                psf_tmp = np.copy(psf_array)
                print(x_pix[i], y_pix[i])
                psf_tmp = spi.shift(psf_tmp, (x_pix[i], y_pix[i]), order=1)
                slice_array += psf_tmp * slice_photons[i]

        elif params["sub_pixel"] == "raw":
            x_int, y_int = np.floor(x_pix), np.floor(y_pix)
            i = (ax + x_int[mask]).astype(int)
            j = (ay + y_int[mask]).astype(int)
            slice_array[i, j] = slice_photons[mask]

        else:
            # If astrometric precision is not that important and everything
            # has been oversampled, use this section.
            #  - ax, ay are the pixel coordinates of the image centre
            # use np.floor instead of int-ing
            x_int, y_int = np.floor(x_pix), np.floor(y_pix)
            i = (ax + x_int[mask]).astype(int)
            j = (ay + y_int[mask]).astype(int)
            for ii, jj, ph in zip(i, j, slice_photons[mask]):
                slice_array[ii, jj] += ph

            try:
                # slice_array = convolve_fft(slice_array, psf.array,
                #                            allow_huge=True)
                # make the move to scipy
                slice_array = fftconvolve(slice_array, psf.array, mode="same")
            except ValueError:
                slice_array = convolve(slice_array, psf.array)

        return slice_array


    def photons_in_range(self, lam_min=None, lam_max=None):
        """

        Number of photons between lam_min and lam_max in units of [ph/s/m2]

        Calculate how many photons for each source exist in the wavelength range
        defined by lam_min and lam_max.

        Parameters
        ----------
        lam_min, lam_max : float, optional
            [um] integrate photons between these two limits. If both are ``None``,
            limits are set at lam[0], lam[-1] for the source's wavelength range

        Returns
        -------
        slice_photons : float
            [ph/s/m2] The number of photons in the wavelength range

        """
        spec_photons = spectrum_sum_over_range(self.lam, self.spectra, lam_min, lam_max)

        slice_photons = spec_photons[self.ref] * self.weight
        return slice_photons


    def scale_spectrum(self, idx=0, mag=20, filter_name="Ks"):
        """
        Scale a certain spectrum to a certain magnitude

        See :func:`simmetis.source.scale_spectrum` for examples

        Parameters
        ----------
        idx : int
            The index of the spectrum to be scaled: <Source>.spectra[idx]
            Default is <Source>.spectra[0]
        mag : float
            [mag] new magnitude of spectrum
        filter_name : str, TransmissionCurve
           Any filter name from SimMETIS or a
           :class:`~.simmetis.spectral.TransmissionCurve` object
           (see :func:`~.simmetis.optics.get_filter_set`)
        """

        self.lam, self.spectra[idx] = scale_spectrum(lam=self.lam,
                                                     spec=self.spectra[idx],
                                                     mag=mag,
                                                     filter_name=filter_name,
                                                     return_ec=False)


    def scale_with_distance(self, distance_factor):
        """
        Scale the source for a new distance

        Scales the positions and brightnesses of the :class:`.Source` object
        according to the ratio of the new and old distances

        i.e. distance_factor = new_distance / current_distance

        .. warning::
            This does not yet take into account redshift

        .. todo::
            Implement redshift

        Parameters
        ----------
        distance_factor : float
            The ratio of the new distance to the current distance
            i.e. distance_factor = new_distance / current_distance

        Examples
        --------
        ::

            >>> from simmetis.source import cluster
            >>>
            >>> curr_dist = 50000  # pc, i.e. LMC
            >>> new_dist = 770000  # pc, i.e. M31
            >>> src = cluster(distance=curr_dist)
            >>> src.scale_with_distance( new_dist/curr_dist )

        """
        self.x /= distance_factor
        self.y /= distance_factor
        self.weight /= distance_factor**2


    def add_background_surface_brightness(self):
        """
        Add an EmissionCurve for the background surface brightness of the object
        """
        pass


    def rotate(self, angle, unit="degree", use_orig_xy=False):
        """
        Rotates the ``x`` and ``y`` coordinates by ``angle`` [degrees]

        Parameters
        ----------
        angle : float
            Default is in degrees, this can set with ``unit``
        unit : str, astropy.Unit
            Either a string with the unit name, or an
            ``astropy.unit.Unit`` object
        use_orig_xy : bool
            If the rotation should be based on the original coordinates or the
            current coordinates (e.g. if rotation has already been applied)

        """
        ang = (angle * u.Unit(unit)).to(u.rad)

        if use_orig_xy:
            xold, yold = self.x_orig, self.y_orig
        else:
            xold, yold = self.x, self.y

        self.x = xold * np.cos(ang) - yold * np.sin(ang)
        self.y = xold * np.sin(ang) + yold * np.cos(ang)


    def shift(self, dx=0, dy=0, use_orig_xy=False):
        """
        Shifts the coordinates of the source by (dx, dy) in [arcsec]

        Parameters
        ----------
        dx, dy : float, array
            [arcsec] The offsets for each coordinate in the arrays ``x``, ``y``.
            - If dx, dy are floats, the same offset is applied to all coordinates
            - If dx, dy are arrays, they must be the same length as ``x``, ``y``
        use_orig_xy : bool
            If the shift should be based on the original coordinates or the
            current coordinates (e.g. if shift has already been applied)

        """
        self.dx = dx
        self.dy = dy

        if use_orig_xy:
            self.x = self.x_orig + dx
            self.y = self.y_orig + dy
        else:
            self.x += dx
            self.y += dy


    def on_grid(self, pix_res=0.004):
        """
        Return an image with the positions of all sources.

        The pixel values correspond to the number of emitting objects in that
        pixel

        Parameters
        ----------
        pix_res : float
            [arcsec] The grid spacing

        Returns
        -------
        im : 2D array
            A numpy array containing an image of where the sources are

        """

        xmin = np.min(self.x)
        ymin = np.min(self.y)
        x_i = ((self.x - xmin) / pix_res).astype(int)
        y_i = ((self.y - ymin) / pix_res).astype(int)
        img = np.zeros((np.max(x_i)+2, np.max(y_i)+2))
        img[x_i, y_i] += 1

        return img


    def read(self, filename):
        """
        Read in a previously saved :class:`.Source` FITS file

        Parameters
        ----------
        filename : str
            Path to the file

        """

        ipt = fits.open(filename)
        dat0 = ipt[0].data
        hdr0 = ipt[0].header
        dat1 = ipt[1].data
        hdr1 = ipt[1].header
        ipt.close()

        self.x = dat0[0, :]
        self.y = dat0[1, :]
        self.ref = dat0[2, :]
        self.weight = dat0[3, :]

        lam_min, lam_max = hdr1["LAM_MIN"], hdr1["LAM_MAX"]
        self.lam_res = hdr1["LAM_RES"]
        self.lam = np.linspace(lam_min, lam_max, hdr1["NAXIS1"])
        self.spectra = dat1

        if "BUNIT" in hdr0.keys():
            self.params["units"] = u.Unit(hdr0["BUNIT"])
        if "EXPTIME" in hdr0.keys():
            self.params["exptime"] = hdr0["EXPTIME"]
        if "AREA"   in hdr0.keys():
            self.params["area"] = hdr0["AREA"]
        if "CDELT1" in hdr0.keys():
            self.params["pix_res"] = hdr0["CDELT1"]
        if "CUNIT1" in hdr0.keys():
            self.params["pix_unit"] = u.Unit(hdr0["CUNIT1"])
        self.lam_res = hdr1["LAM_RES"]

        self._convert_to_photons()


    def write(self, filename):
        """
        Write the current Source object out to a FITS file

        Parameters
        ----------
        filename : str
            where to save the FITS file

        Notes
        -----
        Just a place holder so that I know what's going on with the input table
        * The first extension [0] contains an "image" of size 4 x N where N is the
        number of sources. The 4 columns are x, y, ref, weight.
        * The second extension [1] contains an "image" with the spectra of all
        sources. The image is M x len(spectrum), where M is the number of unique
        spectra in the source list. M = max(ref) - 1
        """

        # hdr = fits.getheader("../../../PreSim/Input_cubes/GC2.fits")
        # ipt = fits.getdata("../../../PreSim/Input_cubes/GC2.fits")
        # flux_map = np.sum(ipt, axis=0).astype(dtype=np.float32)
        # x,y = np.where(flux_map != 0)
        # ref = np.arange(len(x))
        # weight = np.ones(len(x))
        # spectra = np.swapaxes(ipt[:,x,y], 0, 1)
        # lam = np.linspace(0.2,2.5,231)

        xyHDU = fits.PrimaryHDU(np.array((self.x, self.y, self.ref, self.weight)))
        xyHDU.header["X_COL"] = "1"
        xyHDU.header["Y_COL"] = "2"
        xyHDU.header["REF_COL"] = "3"
        xyHDU.header["W_COL"] = "4"

        xyHDU.header["BUNIT"] = self.units.to_string()
        xyHDU.header["EXPTIME"] = self.params["exptime"]
        xyHDU.header["AREA"] = self.params["area"]
        xyHDU.header["CDELT1"] = self.params["pix_res"]
        xyHDU.header["CDELT2"] = self.params["pix_res"]
        xyHDU.header["CUNIT1"] = self.params["pix_unit"]
        xyHDU.header["CUNIT2"] = self.params["pix_unit"]

        xyHDU.header["SIMMETIS"] = "SOURCE"

        specHDU = fits.ImageHDU(self.spectra)
        specHDU.header["CRVAL1"] = self.lam[0]
        specHDU.header["CRPIX1"] = 0
        specHDU.header["CDELT1"] = (self.lam_res, "[um] Spectral resolution")
        specHDU.header["LAM_MIN"] = (self.lam[0], "[um] Minimum wavelength")
        specHDU.header["LAM_MAX"] = (self.lam[-1], "[um] Maximum wavelength")
        specHDU.header["LAM_RES"] = (self.lam_res, "[um] Spectral resolution")

        hdu = fits.HDUList([xyHDU, specHDU])
        hdu.writeto(filename, overwrite=True, checksum=True)


    @property
    def info_keys(self):
        """Return keys of the `info` dict"""
        return self.info.keys()


    def _apply_transmission_curve(self, transmission_curve):
        """
        Apply the values from a TransmissionCurve object to self.spectra

        Parameters
        ----------
        transmission_curve : TransmissionCurve
            The TransmissionCurve to be applied

        See Also
        --------
        :class:`simmetis.spectral.TransmissionCurve`

        """
        tc = deepcopy(transmission_curve)
        tc.resample(self.lam, use_default_lam=False)
        self.spectra = self.spectra_orig * tc.val


    def _convert_to_photons(self):
        """
        Convert the spectra to photons/(s m2)

        If [arcsec] are in the units, we want to find the photons per pixel.
        If [um] are in the units, we want to find the photons per wavelength bin.

        .. todo::
            Come back and put in other energy units like Jy, mag, ergs

        """

        self.units = u.Unit(self.params["units"])
        bases = self.units.bases

        factor = u.Quantity(1.)
        if u.s not in bases:
            factor /= (self.params["exptime"] * u.s)
        if u.m not in bases:
            factor /= (1. * u.m**2)
        if u.micron in bases:
            factor *= (self.lam_res * u.um)
        if u.arcsec in bases:
            factor *= (self.params["pix_res"] * u.arcsec)**2

        self.units = self.units * factor.unit
        self.spectra *= factor.value


    def _from_cube(self, filename):
        # Should this be a class method?
        """
        Make a Source object from a cube in memory or a FITS cube on disk

        Parameters
        ----------
        filename : str
            Path to the FITS cube

        """

        if isinstance(filename, str) and os.path.exists(filename):
            hdr = fits.getheader(filename)
            cube = fits.getdata(filename)
        else:
            raise ValueError(filename + " doesn't exist")

        lam_res = hdr["CDELT3"]
        lam_min = hdr["CRVAL3"] - hdr["CRPIX3"] * lam_res
        lam_max = lam_min + hdr["NAXIS3"] * lam_res

        flux_map = np.sum(cube, axis=0).astype(dtype=np.float32)
        x, y = np.where(flux_map != 0)

        self.lam = np.linspace(lam_min, lam_max, hdr["NAXIS3"])
        self.spectra = np.swapaxes(cube[:, x, y], 0, 1)
        self.x = x
        self.y = y
        self.ref = np.arange(len(x))
        self.weight = np.ones(len(x))

        if "BUNIT" in hdr.keys():
            self.params["units"] = u.Unit(hdr["BUNIT"])
        if "EXPTIME" in hdr.keys():
            self.params["exptime"] = hdr["EXPTIME"]
        if "AREA"   in hdr.keys():
            self.params["area"] = hdr["AREA"]
        if "CDELT1" in hdr.keys():
            self.params["pix_res"] = hdr["CDELT1"]
        if "CUNIT1" in hdr.keys():
            self.params["pix_unit"] = hdr["CUNIT1"]
        self.lam_res = lam_res

        self._convert_to_photons()


    def _from_arrays(self, lam, spectra, x, y, ref, weight=None):
        # Should this be a class method?
        """
        Make a Source object from a series of lists

        Parameters
        ----------
        lam : np.ndarray
            Dimensions (1, m) with m spectral bins
        spectra : np.ndarray
            Dimensions (n, m) for n SEDs, each with m spectral bins
        x, y : np.ndarray
            [arcsec] each (1, n) for the coordinates of n emitting objects
        ref : np.ndarray
            Dimensions (1, n) for referencing each coordinate to a spectrum
        weight : np.ndarray, optional
            Dimensions (1, n) for weighting the spectrum of each object

        """

        self.lam = lam
        self.spectra = spectra
        self.x = x
        self.y = y
        self.ref = ref
        if weight is not None:
            self.weight = weight
        else:
            self.weight = np.array([1] * len(x))
        self.lam_res = np.median(lam[1:] - lam[:-1])

        if len(spectra.shape) == 1:
            self.spectra = np.array([spectra])

        self._convert_to_photons()


    def __str__(self):
        return "A photon source object"


    def __array__(self):
        if self.array is None:
            return np.zeros((self.naxis1, self.naxis2))
        else:
            return self.array


    def __getitem__(self, i):
        return (self.x[i], self.y[i],
                self.spectra[self.ref[i], :] * self.weight[i])


    def __mul__(self, x):
        newsrc = deepcopy(self)
        if isinstance(x, (TransmissionCurve, EmissionCurve,
                          UnityCurve, BlackbodyCurve)):
            newsrc._apply_transmission_curve(x)
        else:
            newsrc.array *= x
        return newsrc


    def __add__(self, x):
        newsrc = deepcopy(self)
        if isinstance(x, Source):
            if self.units != x.units:
                raise ValueError("units are not compatible: " + \
                                 str(self.units) + ", " + str(x.units))

            newsrc.lam = self.lam
            newsrc.spectra = list(self.spectra)
            # Resample new spectra to wavelength grid of self
            for spec in x.spectra:
                tmp = np.interp(self.lam, x.lam, spec)
                scale_factor = np.sum(spec) / np.sum(tmp)
                newsrc.spectra += [tmp * scale_factor]
            newsrc.spectra = np.asarray(newsrc.spectra)
            newsrc.spectra_orig = newsrc.spectra
            newsrc.x = np.array((list(self.x) + list(x.x)))
            newsrc.y = np.array((list(self.y) + list(x.y)))
            newsrc.ref = np.array((list(self.ref) + list(x.ref + self.spectra.shape[0])))
            newsrc.weight = np.array((list(self.weight) + list(x.weight)))

            newsrc.x_orig = deepcopy(newsrc.x)
            newsrc.y_orig = deepcopy(newsrc.y)

        else:
            newsrc.array += x

        newsrc.info["object"] = "combined"

        return newsrc


    def __sub__(self, x):
        newsrc = deepcopy(self)
        newsrc.array -= x
        return newsrc


    def __rmul__(self, x):
        return self.__mul__(x)


    def __radd__(self, x):
        return self.__add__(x)


    def __rsub__(self, x):
        return self.__mul__(-1) + x


    def __imul__(self, x):
        return self.__mul__(x)


    def __iadd__(self, x):
        return self.__add__(x)


    def __isub__(self, x):
        return self.__sub__(x)


##############################################################################




def _get_stellar_properties(spec_type, cat=None, verbose=False):
    """
    Returns an :class:`astropy.Table` with the list of properties for the
    star(s) in ``spec_type``

    Parameters
    ----------
    spec_type : str, list
        The single or list of spectral types
    cat : str, optional
        The filename of a catalogue in a format readable by
        :func:`astropy.io.ascii.read`, e.g. ASCII, CSV. The catalogue should
        contain stellar properties
    verbose : bool
        Print which stellar type is being considered

    Returns
    -------
    props : :class:`astropy.Table` or list of :class:`astropy.Table` objects
        with stellar parameters

    """

    if cat is None:
        cat = ioascii.read(os.path.join(__pkg_dir__, "data",
                                        "EC_all_stars.csv"))

    if isinstance(spec_type, (list, tuple)):
        return [_get_stellar_properties(i, cat) for i in spec_type]
    else:
        # Check if stellar type is in cat; if not look for the next
        # type in the sequence that is and assign its values
        spt, cls, lum = spec_type[0], int(spec_type[1]), spec_type[2:]
        for _ in range(10):
            if cls > 9:
                cls = 0
                spt = "OBAFGKMLT"["OBAFGKMLT".index(spt)+1]

            startype = spt+str(cls)+lum # was 'star', redefined function star()
            cls += 1

            if startype in cat["Stellar_Type"]:
                break

        else:   # for loop did not find anything
            raise ValueError(spec_type+" doesn't exist in the database")

        n = np.where(cat["Stellar_Type"] == startype.upper())[0][0]
        if verbose:
            print("Returning properties for", startype)

        return cat[n]


def _get_stellar_mass(spec_type):
    """
    Returns a single (or list of) float(s) with the stellar mass(es)

    Parameters
    ----------
    spec_type : str, list
        The single or list of spectral types in the normal format: G2V

    Returns
    -------
    mass : float, list
        [Msol]

    """

    props = _get_stellar_properties(spec_type)

    if isinstance(props, (list, tuple)):
        return [prop["Mass"] for prop in props]
    else:
        return props["Mass"]


def _get_stellar_Mv(spec_type):
    """
    Returns a single (or list of) float(s) with the V-band absolute magnitude(s)

    Parameters
    ----------
    spec_type : str, list
        The single or list of spectral types

    Returns
    -------
    Mv : float, list

    """

    props = _get_stellar_properties(spec_type)

    if isinstance(props, (list, tuple)):
        return [prop["Mv"] for prop in props]
    else:
        return props["Mv"]







def _get_pickles_curve(spec_type, cat=None, verbose=False):
    """
    Returns the emission curve for a single or list of ``spec_type``, normalised
    to 5556A

    Parameters
    ----------
    spec_type : str, list
        The single (or list) of spectral types (i.e. "A0V" or ["K5III", "B5I"])

    Returns
    -------
    lam : np.array
        a single np.ndarray for the wavelength bins of the spectrum,
    val : np.array (list)
        a (list of) np.ndarray for the emission curve of the spectral type(s)
        relative to the flux at 5556A

    References
    ----------
    Pickles 1998 - DOI: 10.1086/316197

    """
    if cat is None:
        cat = fits.getdata(os.path.join(__pkg_dir__, "data", "EC_pickles.fits"))

    if isinstance(spec_type, (list, tuple)):
        return cat["lam"], [_get_pickles_curve(i, cat)[1] for i in spec_type]
    else:
        # split the spectral type into 3 components and generalise for Pickles
        spt, cls, lum = spec_type[0], int(spec_type[1]), spec_type[2:]
        if lum.upper() == "I":
            lum = "Ia"
        elif lum.upper() == "II":
            lum = "III"
        elif "V" in lum.upper():
            lum = "V"

        for _ in range(10):  # TODO: What does this loop do? (OC)
            if cls > 9:
                cls = 0
                spt = "OBAFGKMLT"["OBAFGKMLT".index(spt)+1]
            startype = spt + str(cls) + lum
            cls += 1

            if startype in cat.columns.names:
                break

        if spec_type != startype and verbose:
            print(spec_type, "isn't in Pickles. Returned", startype)

        try:
            lam, spec = cat["lam"], cat[startype]
        except KeyError:      # Correct? This shouldn't use error handling.
            lam, spec = cat["lam"], cat["M9III"]
        return lam, spec


def _scale_pickles_to_photons(spec_type, mag=0):
    """
    Pull in a spectrum from the Pickles library and scale to V=0 star

    Parameters
    ----------
    spec_type : str, list
        A (list of) spectral type(s), e.g. "A0V" or ["A0V", G2V"]
    mag : float, list, optional
        A (list of) magnitudes for the spectral type(s). Default is 0

    Returns
    -------
    lam, ec : array
        The wavelength bins and the SEDs for the spectral type

    Notes
    -----
    - Vega has a 5556 flux of between 950 and 1000 ph/s/cm2/A. The pickles
    resolution is 5 Ang.
    - Therefore the flux at 5555 should be 5 * 1000 * 10^(-0.4*Mv) ph/s/cm2/bin
    - Pickles catalogue is in units of Flambda [erg/s/cm2/A]
    - Ergo we need to divide the pickels values by lam/0.5556[nm], then rescale
    Regarding the number of photons in the 1 Ang bin at 5556 Ang
    - Bohlin (2014) says F(5556)=3.44×10−9 erg cm−2 s−1 A−1
    - Values range from 3.39 to 3.46 with the majority in range 3.44 to 3.46.
      Bohlin recommends 3.44
    - This results in a photon flux of 962 ph cm-2 s-1 A-1 at 5556 Ang

    """

    if isinstance(spec_type, (list, tuple, np.ndarray)):
        if isinstance(mag, (list, tuple, np.ndarray)):
            if len(mag) != len(spec_type):
                raise ValueError("len(mag) != len(spec_type)")
            mag = list(mag)
        else:
            mag = [mag]*len(spec_type)
    else:
        mag = [mag]

    mag = np.asarray(mag)

    Mv = _get_stellar_Mv(spec_type)
    if not hasattr(Mv, "__len__"):
        Mv = [Mv]

    Mv = np.asarray(Mv)
    lam, ec = _get_pickles_curve(spec_type)
    dlam = (lam[1:] - lam[:-1])
    dlam = np.append(dlam, dlam[-1])

    lam *= 1E-4         # convert to um from Ang

    # Use Bohlin (2014) to determine the photon flux of a mag 0 A0V star
    # at 5556 Ang
    F = 3.44E-9 * u.erg / (u.cm**2 * u.s * u.AA)
    E = c.c*c.h/(5556*u.AA)
    ph0 = (F/E).to(1/(u.s * u.cm**2 * u.AA)).value

    # 5 Ang/bin * ~962 ph/s * (abs mag + apparent mag)

    ph_factor = []
    for i in range(len(mag)):
        tmp = dlam * ph0 * 10**(-0.4*(Mv[i] + mag[i]))
        ph_factor += [tmp]

    # take care of the conversion to ph/s/m2 by multiplying by 1E4
    # TODO: The original type(ec) == (list, tuple) is wrong (should be 'in')
    #   However, correcting it (using idiomatic isinstance) breaks the code!
    #   There must be a bug.
    # Correct code:
    # if isinstance(ec, (list, tuple)):
    #     for i in range(len(ec)):
    if type(ec) == (list, tuple):
        for i in len(range(ec)):
            ec[i] *= (lam/0.5556) * ph_factor[i] * 1E4
    else:
        ec *= (lam/0.5556) * ph_factor[0] * 1E4

    return lam, ec


def BV_to_spec_type(B_V):
    """
    Returns the latest main sequence spectral type(s) for (a) B-V colour

    Parameters
    ----------
    B_V : float, array
        [mag] B-V colour

    Returns
    -------
    spec_types : list
        A list of the spectral types corresponding to the B-V colours

    Examples
    --------
    ::

        >>> BV = np.arange(-0.3, 2.5, 0.5)
        >>> spec_types = BV_to_spec_type(BV)
        >>> print(BV)
        >>> print(spec_types)
        [-0.3  0.2  0.7  1.2  1.7  2.2]
        ['O9V', 'A8V', 'G2V', 'K5V', 'M3V', 'M8V']

    """

    #from simmetis.source import _get_stellar_properties

    spec_type = [spt+str(i)+"V" for spt in "OBAFGKM" for i in range(10)]
    B_V_int = np.array([spt["B-V"] for spt in _get_stellar_properties(spec_type)])

    idx = np.round(np.interp(B_V, B_V_int, np.arange(len(B_V_int)))).astype(int)
    if np.isscalar(idx):
        idx = np.array([idx])
    spec_types = [spec_type[i] for i in idx]

    return spec_types


def mag_to_photons(filter_name, magnitude=0):
    """
    Return the number of photons for a certain filter and magnitude

    Parameters
    ----------
    filter_name : str
        filter name. See simmetis.optics.get_filter_set()
    magnitude : float
        [mag] the source brightness

    Returns
    -------
    flux : float
        [ph/s/m2] Photon flux in the given filter

    See Also
    --------
    :func:`.photons_to_mag`
    :func:`.zero_magnitude_photon_flux`,
    :func:`simmetis.optics.get_filter_set`
    """

    flux_0 = zero_magnitude_photon_flux(filter_name)
    flux = flux_0 * 10**(-0.4 * magnitude)
    return flux


def photons_to_mag(filter_name, photons=1):
    """
    Return the number of photons for a certain filter and magnitude

    Parameters
    ----------
    filter_name : str
        filter name. See simmetis.optics.get_filter_set()
    photons : float
        [ph/s/m2] the integrated photon flux for the filter

    Returns
    -------
    mag : float
        The magnitude of an object with the given photon flux through the filter

    See Also
    --------
    :func:`.photons_to_mag`
    :func:`.zero_magnitude_photon_flux`,
    :func:`simmetis.optics.get_filter_set`

    """

    flux_0 = zero_magnitude_photon_flux(filter_name)
    mag = -2.5 * np.log10(photons / flux_0)
    return mag



def _get_refstar_curve(filename=None,mag=0):
    """
    """
    ## TODO: Can we pre-select a star based on the instrument we're simulating?
    ##       Do we need more flexibility in the path?
    #data = ioascii.read(os.path.join(__pkg_dir__, "data", "vega.dat"))
    data = ioascii.read(os.path.join(__pkg_dir__, "data",
                                     "sirius_downsampled.txt"))

    mag_scale_factor = 10**(-mag/2.5)

    ##
    ## this function is expected to return the number of photons of a 0th mag star
    ## for a star brighter than 0th mag, the number of photons needs to be reduced to match a 0th mag star
    lam, spec = data[data.colnames[0]], data[data.colnames[1]]/mag_scale_factor
    return lam, spec



def zero_magnitude_photon_flux(filter_name):
    """
    Return the number of photons for a m=0 star for a certain filter

    Parameters
    ----------
    filter_name : str
        filter name. See simmetis.optics.get_filter_set()

    Notes
    -----
    units in [ph/s/m2]
    """

    if isinstance(filter_name, TransmissionCurve):
        vlam = filter_name.lam
        vval = filter_name.val

    else:
        if os.path.exists(filter_name):
            fname = filter_name
        elif os.path.exists(os.path.join(__pkg_dir__, "data", filter_name)):
            fname = os.path.join(__pkg_dir__, "data", filter_name)
        elif os.path.exists(__pkg_dir__, "data",
                                 "TC_filter_" + filter_name + ".dat"):
            fname = os.path.join(__pkg_dir__, "data",
                                 "TC_filter_" + filter_name + ".dat")
        else:
                raise ValueError("File " + fname + " does not exist")

        vraw = ioascii.read(fname)
        vlam = vraw[vraw.colnames[0]]
        vval = vraw[vraw.colnames[1]]

    #lam, vega = _scale_pickles_to_photons("A0V", mag=-0.58)
    ##
    ## we refer here (SimMETIS) to the Sirius spectrum (see _get_refstar_curve above)
    ## and give the Vega magnitude of Sirius in L/M band.
    lam, vega = _get_refstar_curve(mag=-1.39)
    filt = np.interp(lam, vlam, vval)

    n_ph = np.sum(vega*filt)

    #print("units in [ph/s/m2]")
    return n_ph


def value_at_lambda(lam_i, lam, val, return_index=False):
    """
    Return the value at a certain wavelength - i.e. val[lam] = x

    Parameters
    ----------
    lam_i : float
        the wavelength of interest
    lam : np.ndarray
        an array of wavelengths
    val : np.ndarray
        an array of values
    return_index : bool, optional
        If True, the index of the wavelength of interest is returned
        Default is False
    """

    i0 = np.where(lam <= lam_i)[0][-1]
    i1 = np.where(lam > lam_i)[0][0]

    lam_x = np.array([lam[i0], lam_i, lam[i1]])
    val_i = np.interp(lam_x, lam, val)

    if return_index:
        return i0
    else:
        return val_i[1]


def get_SED_names(path=None):
    """
    Return a list of the SEDs installed in the package directory

    Looks for files that follow the naming convention ``SED_<name>.dat``.
    For example, SimMETIS contains an SED for an elliptical galaxy named
    ``SED_elliptical.dat``

    Parameters
    ----------
    path : str, optional
        Directory to look in for filters

    Returns
    -------
    sed_names : list
        A list of names for the SED files available

    Examples
    --------
    Names returned here can be used with the function :func:`.SED` to call up
    ::

        >>> from simmetis import SED, get_SED_names
        >>> print(get_SED_names())
        ['elliptical', 'interacting', 'spiral', 'starburst', 'ulirg']
        >>> SED("spiral")
        (array([ 0.3  ,  0.301,  0.302, ...,  2.997,  2.998,  2.999]),
         array([        0.        ,         0.        ,  26055075.98709349, ...,
                  5007498.76444208,   5000699.21993188,   4993899.67542169]))

    See Also
    --------
    :func:`.SED`

    """
    if path is None:
        path = os.path.join(__pkg_dir__, "data")
    sed_names = [i.replace(".dat", "").split("SED_")[-1] \
                                for i in glob(os.path.join(path, "SED_*.dat"))]

    sed_names += ["All stellar spectral types (e.g. G2V, K0III)"]
    return sed_names




def SED(spec_type, filter_name="V", magnitude=0.):
    """
    Return a scaled SED for a star or type of galaxy

    The SED can be for stellar spectra of galacty spectra. It is best not to mix
    the two types when calling ``SED()``. Either provide a list of stellar types,
    e.g. ["G2V", "A0V"], of a list of galaxy types, e.g. ["elliptical", "starburst"]

    To get the list of galaxy types that are installed, call get_SED_names().
    All stellar types from the Pickles (1998) catalogue are available.

    Parameters
    ----------
    spec_type : str, list
        The spectral type of the star(s) - from the Pickles 1998 catalogue
        The names of a galaxy spectrum - see get_SED_names()
    filter_name : str, optional
        Default is "V". Any filter in the simmetis/data directory can be used,
        or the user can specify a file path to an ASCII file for the filter
    magnitude : float, list, optional
        Apparent magnitude of the star. Default is 0.

    Returns
    -------
    lam : np.ndarray
        [um] The centre of each 5 Ang bin along the spectral axis
    val : np.ndarray
        [ph/s/m2/bin] The photon flux of the star in each bin


    Examples
    --------

    Get the SED and the wavelength bins for a J=0 A0V star

        >>> from simmetis.source import SED
        >>> lam, spec = SED("A0V", "J", 0)

    Get the SED for a generic starburst galaxy

        >>> lam, spec = SED("starburst")

    Get the SEDs for several spectral types with different magnitudes

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt
        from simmetis.source import SED

        lam, spec = SED(spec_type=["A0V", "G2V"],
                            filter_name="PaBeta",
                            magnitude=[15, 20])

        plt.plot(lam, spec[0], "blue", label="Vega")
        plt.plot(lam, spec[1], "orange", label="G2V")
        plt.semilogy(); plt.legend(); plt.show()



    Notes
    -----
    Original flux units for the stellar spectra are in [ph/s/m2/AA], so we
    multiply the flux by 5 to get [ph/s/m2/bin]. Therefore divide by 5*1E4 if
    you need the flux in [ph/s/cm2/Angstrom]

    """

    if isinstance(spec_type, (tuple, list, np.ndarray)):
        spec_type = list(spec_type)
        if np.isscalar(magnitude):
            magnitude = [magnitude]*len(spec_type)
    elif isinstance(spec_type, str):
        spec_type = [spec_type]

    if isinstance(magnitude, (list, tuple)):
        magnitude = np.asarray(magnitude)

    # Check if any of the names given are in the package directory
    gal_seds = get_SED_names()
    if np.any([i in gal_seds for i in spec_type]):
        galflux = []
        for gal in spec_type:
            data = ioascii.read(__pkg_dir__+"/data/SED_"+gal+".dat")
            galflux += [data[data.colnames[1]]]
            galflux = np.asarray(galflux)
        lam = data[data.colnames[0]]

        lam, galflux = scale_spectrum(lam=lam, spec=galflux, mag=magnitude,
                                      filter_name=filter_name)
        return lam, galflux

    else:
        lam, starflux = _scale_pickles_to_photons(spec_type)
        lam, starflux = scale_spectrum(lam=lam, spec=starflux, mag=magnitude,
                                       filter_name=filter_name)

        return lam, starflux


def empty_sky():
    """
    Returns an empty source so that instrumental fluxes can be simulated

    Returns
    -------
    sky : Source

    """
    sky = Source(lam=np.linspace(0.3, 3.0, 271),
                 spectra=np.zeros((1, 271)),
                 x=[0], y=[0], ref=[0], weight=[0])
    return sky


def star_grid(n, mag_min, mag_max, filter_name="Ks", separation=1,
              spec_type="A0V"):
    """
    Creates a square grid of A0V stars at equal magnitude intervals

    Parameters
    ----------
    n : float
        the number of stars in the grid
    mag_min, mag_max : float
        [vega mag] the minimum (brightest) and maximum (faintest) magnitudes for
        stars in the grid
    filter_name : str
        any filter that is in the SimMETIS package directory.
        See ``simmetis.optics.get_filter_set()``
    separation : float, optional
        [arcsec] an average speration between the stars in the grid can be
        specified. Default is 1 arcsec
    spec_type : str, optional
        the spectral type of the star, e.g. "A0V", "G5III"

    Returns
    -------
    source : ``simmetis.Source``

    Notes
    -----
    The units of the A0V spectrum in ``source`` are [ph/s/m2/bin].
    The weight values are the scaling factors to bring a V=0 A0V spectrum down
    to the required magnitude for each star.

    """

    if isinstance(mag_min, (list, tuple, np.ndarray)):
        mags = np.asarray(mag_min)
    else:
        if mag_min < mag_max:
            mags = np.linspace(mag_min, mag_max, n)
        elif mag_min > mag_max:
            mags = np.linspace(mag_max, mag_min, n)
        elif mag_min == mag_max:
            mags = np.ones(n) * mag_min

    side_len = int(np.sqrt(n)) + (np.sqrt(n) % 1 > 0)

    x = separation * (np.arange(n) % side_len - (side_len - 1) / 2)
    y = separation * (np.arange(n)// side_len - (side_len - 1) / 2)

    lam, spec = SED(spec_type, filter_name=filter_name, magnitude=0)
    if isinstance(spec_type, (list, tuple)):
        ref = np.arange(len(spec_type))
    else:
        ref = np.zeros((n))
    weight = 10**(-0.4*mags)

    units = "ph/s/m2"

    src = Source(lam=lam, spectra=spec,
                 x=x, y=y,
                 ref=ref, weight=weight,
                 units=units)

    return src



def star(spec_type="A0V", mag=0, filter_name="Ks", x=0, y=0, **kwargs):
    """
    Creates a simmetis.Source object for a star with a given magnitude

    This is just the single star variant for ``simmetis.source.stars()``

    Parameters
    ----------
    spec_type : str
        the spectral type of the star, e.g. "A0V", "G5III"
    mag : float
        magnitude of star
    filter_name : str
        Filter in which the magnitude is given. Can be the name of any filter
        curve file in the simmetis/data folder, or a path to a custom ASCII file
    x, y : float, int, optional
        [arcsec] the x,y position of the star on the focal plane


    Keyword arguments
    -----------------
    Passed to the ``simmetis.Source`` object. See the docstring for this object.

    pix_unit : str
        Default is "arcsec". Acceptable are "arcsec", "arcmin", "deg", "pixel"
    pix_res : float
        [arcsec] The pixel resolution of the detector. Useful for surface
        brightness calculations

    Returns
    -------
    source : ``simmetis.Source``

    See Also
    --------
    .stars()

    """

    thestar = stars([spec_type], [mag], filter_name, [x], [y], **kwargs)
    return thestar


def stars(spec_types=("A0V"), mags=(0), filter_name="Ks",
          x=None, y=None, **kwargs):
    """
    Creates a simmetis.Source object for a bunch of stars.

    Parameters
    ----------
    spec_types : str, list of strings
        the spectral type(s) of the stars, e.g. "A0V", "G5III"
        Default is "A0V"
    mags : float, array
        [mag] magnitudes of the stars.
    filter_name : str,
        Filter in which the magnitude is given. Can be the name of any filter
        curve file in the simmetis/data folder, or a path to a custom ASCII file
    x, y : arrays
        [arcsec] x and y coordinates of the stars on the focal plane


    Keyword arguments
    -----------------
    Passed to the ``simmetis.Source`` object. See the docstring for this object.

    pix_unit : str
        Default is "arcsec". Acceptable are "arcsec", "arcmin", "deg", "pixel"
    pix_res : float
        [arcsec] The pixel resolution of the detector. Useful for surface
        brightness calculations

    Returns
    -------
    source : ``simmetis.Source``


    Examples
    --------

    Create a ``Source`` object for a random group of stars

        >>> import numpy as np
        >>> from simmetis.source import stars
        >>>
        >>> spec_types = ["A0V", "G2V", "K0III", "M5III", "O8I"]
        >>> ids = np.random.randint(0,5, size=100)
        >>> star_list = [spec_type[i] for i in ids]
        >>> mags = np.random.normal(20, 3, size=100)
        >>>
        >>> src = stars(spec_types, mags, filter_name="Ks)

    If we don't specify any coordinates all stars have the position (0, 0).
    **All positions are in arcsec.**
    There are two possible ways to add positions. If we know them to begin with
    we can add them when generating the source full of stars

        >>> x, y = np.random.random(-20, 20, size=(100,2)).tolist()
        >>> src = stars(star_list, mags, filter_name="Ks, x=x, y=y)

    Or we can add them to the ``Source`` object directly (although, there are
    less checks to make sure the dimensions match here):

        >>> src.x, src.y = x, y


    """

    if isinstance(spec_types, str):
        spec_types = [spec_types]

    if isinstance(mags, (int, float)):
        mags = [mags] * len(spec_types)

    if len(mags) > 1  and len(spec_types) == 1:
        spec_types *= len(mags)
    elif len(mags) != len(spec_types):
        raise ValueError("len(mags) != len(spec_types)")

    mags = np.array(mags)

    if x is None:
        x = np.zeros(len(mags))
    if y is None:
        y = np.zeros(len(mags))

    # only pull in the spectra for unique spectral types

    # assign absolute magnitudes to stellar types in cluster
    unique_types = np.unique(spec_types)
    lam, spec = SED(unique_types, filter_name=filter_name, magnitude=[0]*len(unique_types))

    # get the references to the unique stellar types
    ref_dict = {i : j for i, j in zip(unique_types, np.arange(len(unique_types)))}
    if isinstance(spec_types, (list, tuple, np.ndarray)):
        ref = np.array([ref_dict[i] for i in spec_types])
    else:
        ref = np.zeros(len(mags))

    weight = 10**(-0.4*mags)

    units = "ph/s/m2"

    src = Source(lam=lam, spectra=spec,
                 x=x, y=y,
                 ref=ref, weight=weight,
                 units=units, **kwargs)

    src.info["object"] = "stars"
    src.info["spec_types"] = spec_types
    src.info["magnitudes"] = mags
    src.info["filter_name"] = filter_name

    return src


def source_1E4_Msun_cluster(distance=50000, half_light_radius=1):
    """
    Generate a source object for a 10^4 solar mass cluster

    Parameters
    ----------
    distance : float
        [pc] distance to the cluster
    half_light_radius : float
        [pc] half light radius of the cluster
    mass : float
        [Msun] If you'd like a different size cluster

    Returns
    -------
    src : simmetis.Source

    See Also
    --------
    .cluster()

    """
    # IMF is a realisation of stellar masses drawn from an initial mass
    # function (TODO: which one?) summing to 1e4 M_sol.
    fname = os.path.join(__pkg_dir__, "data", "IMF_1E4.dat")
    imf = np.loadtxt(fname)

    # Assign stellar types to the masses in imf using list of average
    # main-sequence star masses:
    stel_type = [i + str(j) + "V" for i in "OBAFGKM" for j in range(10)]
    mass = _get_stellar_mass(stel_type)
    ref = utils.nearest(mass, imf)
    thestars = [stel_type[i] for i in ref] # was stars, redefined function name

    # assign absolute magnitudes to stellar types in cluster
    unique_ref = np.unique(ref)
    unique_type = [stel_type[i] for i in unique_ref]
    unique_Mv = _get_stellar_Mv(unique_type)

    # Mv_dict = {i : float(str(j)[:6]) for i, j in zip(unique_type, unique_Mv)}
    ref_dict = {i: j for i, j in zip(unique_type, np.arange(len(unique_type)))}

    # find spectra for the stellar types in cluster
    lam, spectra = _scale_pickles_to_photons(unique_type)

    # this one connects the stars to one of the unique spectra
    stars_spec_ref = [ref_dict[i] for i in thestars]

    # absolute mag + distance modulus
    m = np.array([unique_Mv[i] for i in stars_spec_ref])
    m += 5 * np.log10(distance) - 5

    # set the weighting
    weight = 10**(-0.4*m)

    # draw positions of stars: cluster has Gaussian profile
    distance *= u.pc
    half_light_radius *= u.pc
    hwhm = (half_light_radius/distance*u.rad).to(u.arcsec).value
    sig = hwhm / np.sqrt(2 * np.log(2))

    x = np.random.normal(0, sig, len(imf))
    y = np.random.normal(0, sig, len(imf))

    src = Source(lam=lam, spectra=spectra, x=x, y=y, ref=stars_spec_ref,
                 weight=weight, units="ph/s/m2")

    return src


def cluster(mass=1E3, distance=50000, half_light_radius=1):
    """
    Generate a source object for a cluster

    The cluster distribution follows a gaussian profile with the
    ``half_light_radius`` corresponding to the HWHM of the distribution. The
    choice of stars follows a Kroupa IMF, with no evolved stars in the mix. Ergo
    this is more suitable for a young cluster than an evolved custer

    Parameters
    ----------
    mass : float
        [Msun] Mass of the cluster (not number of stars). Max = 1E5 Msun
    distance : float
        [pc] distance to the cluster
    half_light_radius : float
        [pc] half light radius of the cluster

    Returns
    -------
    src : simmetis.Source

    Examples
    --------

    Create a ``Source`` object for a young open cluster with half light radius
    of around 0.2 pc at the galactic centre and 100 solar masses worth of stars:

        >>> from simmetis.source import cluster
        >>> src = cluster(mass=100, distance=8500, half_light_radius=0.2)


    """
    # IMF is a realisation of stellar masses drawn from an initial mass
    # function (TODO: which one?) summing to 1e4 M_sol.
    if mass <= 1E4:
        fname = os.path.join(__pkg_dir__, "data", "IMF_1E4.dat")
        imf = np.loadtxt(fname)
        imf = imf[0:int(mass/1E4 * len(imf))]
    elif mass > 1E4 and mass < 1E5:
        fname = os.path.join(__pkg_dir__, "data", "IMF_1E5.dat")
        imf = np.loadtxt(fname)
        imf = imf[0:int(mass/1E5 * len(imf))]
    else:
        raise ValueError("Mass too high. Must be <10^5 Msun")

    # Assign stellar types to the masses in imf using list of average
    # main-sequence star masses:
    stel_type = [i + str(j) + "V" for i in "OBAFGKM" for j in range(10)]
    masses = _get_stellar_mass(stel_type)
    ref = utils.nearest(masses, imf)
    thestars = [stel_type[i] for i in ref] # was stars, redefined function name

    # assign absolute magnitudes to stellar types in cluster
    unique_ref = np.unique(ref)
    unique_type = [stel_type[i] for i in unique_ref]
    unique_Mv = _get_stellar_Mv(unique_type)

    # Mv_dict = {i : float(str(j)[:6]) for i, j in zip(unique_type, unique_Mv)}
    ref_dict = {i : j for i, j in zip(unique_type, np.arange(len(unique_type)))}

    # find spectra for the stellar types in cluster
    lam, spectra = _scale_pickles_to_photons(unique_type)

    # this one connects the stars to one of the unique spectra
    stars_spec_ref = [ref_dict[i] for i in thestars]

    # absolute mag + distance modulus
    m = np.array([unique_Mv[i] for i in stars_spec_ref])
    m += 5 * np.log10(distance) - 5

    # set the weighting
    weight = 10**(-0.4*m)

    # draw positions of stars: cluster has Gaussian profile
    distance *= u.pc
    half_light_radius *= u.pc
    hwhm = (half_light_radius/distance*u.rad).to(u.arcsec).value
    sig = hwhm / np.sqrt(2 * np.log(2))

    x = np.random.normal(0, sig, len(imf))
    y = np.random.normal(0, sig, len(imf))

    src = Source(lam=lam, spectra=spectra, x=x, y=y, ref=stars_spec_ref,
                 weight=weight, units="ph/s/m2")

    src.info["object"] = "cluster"
    src.info["total_mass"] = mass
    src.info["masses"] = imf
    src.info["half_light_radius"] = half_light_radius
    src.info["hwhm"] = hwhm
    src.info["distance"] = distance
    src.info["stel_type"] = stel_type

    return src



def source_from_image(images, lam, spectra, plate_scale, oversample=1,
                      units="ph/s/m2", flux_threshold=0,
                      center_offset=(0, 0),
                      conserve_flux=True,
                      **kwargs):
    """
    Create a Source object from an image or a list of images.

    .. note::
        ``plate_scale`` is the original plate scale of the images. If this is
        not the same as the plate scale of the ``Detector`` (i.e. 4mas for MICADO)
        then you will need to specify oversample to interpolate between the two
        scales. I.e.  oversample = Image plate scale / Detector plate scale


    Parameters
    ----------
    images : np.ndarray, list
        A single or list of np.ndarrays describing where the flux is coming from.
        The spectrum for each pixel in the image is weighted by the pixel value.
    lam : np.ndarray
        An array contains the centres of the wavelength bins for the spectra
    spectra : np.ndarray
        A (n,m) array with n spectra, each with m bins
    plate_scale : float
        [arcsec] The plate scale of the images in arcseconds (e.g. 0.004"/pixel)
    oversample : int
        The factor with which to oversample the image. Each image pixel is split
        into (oversample)^2 individual point sources.
    units : str, optional
        The energy units of the spectra. Default is [ph/s/m2]
    flux_threshold : float, optional
        If there is noise in the image, set threshold to the noise limit so that
        only real photon sources are extracted. Default is 0.
    center_offset : (float, float)
        [arcsec] If the centre of the image is offset, add this offset to (x,y)
        coordinates.
    conserve_flux : bool, optional
        If True, when the image is rescaled, flux is conserved
        i.e. np.sum(image) remains constant
        If False, the maximum value of the image stays constant after rescaling
        i.e. np.max(image) remains constant


    Keyword arguments
    -----------------
    Passed to the ``simmetis.Source`` object. See the docstring for this object.

    pix_unit : str
        Default is "arcsec". Acceptable are "arcsec", "arcmin", "deg", "pixel"
    pix_res : float
        [arcsec] The pixel resolution of the detector. Useful for surface
        brightness calculations


    Returns
    -------
    src : source.Source object


    Examples
    --------

    To create a ``Source`` object we need an image that describes the spatial
    distribution of the object of interest and spectrum. For the sake of ease we
    will assign a generic elliptical galaxy spectrum to the image.

        >>> from astropy.io import fits
        >>> from simmetis.source import SED, source_from_image

        >>> im = fits.getdata("galaxy.fits")
        >>> lam, spec = SED("elliptical")
        >>> src = source_from_image(im, lam, spec,
                                    plate_scale=0.004)

    **Note** Here we have assumed that the plate scale of the image is the same
    as the MICADO wide-field mode, i.e. 0.004 arcseconds. If the image is from a
    real observation, or it was generated with a different pixel scale, we will
    need to tell SimMETIS about this:

        >>> src = source_from_image(im, lam, spec,
                                    plate_scale=0.01,
                                    oversample=2.5)

    If the image is from real observations, chances are good that the background
    flux is higher than zero. We can set a ``threshold`` in order to tell
    SimMETIS to ignore all pixel with values below the background level:

        >>> src = source_from_image(im, lam, spec,
                                    plate_scale=0.01,
                                    oversample=2.5,
                                    flux_threshold=0.2)

    Finally, if the image centre is not the centre of the observation, we can
    shift the image relative to the MICADO field of view. The units for the
    offset are [arcsec]

        >>> src = source_from_image(im, lam, spec,
                                    plate_scale=0.01,
                                    oversample=2.5,
                                    flux_threshold=0.2,
                                    center_offset=(10,-15))


    """

    if isinstance(images, (list, tuple)):
        srclist = [source_from_image(images[i], lam, spectra[i, :], plate_scale,
                                     oversample, units, flux_threshold,
                                     center_offset)
                   for i in range(len(images))]
        fullsrc = srclist[0]
        for src in srclist[1:]:
            fullsrc += src
        return fullsrc

    else:
        #if not isinstance(oversample, int):
        #    raise ValueError("Oversample must be of type 'int'")

        if isinstance(images, str) and images.split(".")[-1].lower() == "fits":
            images = fits.getdata(images)

        # im = images
        # y_cen, x_cen = np.array(im.shape) / 2 + np.array(center_offset)
        # # x_cen, y_cen = np.array(im.shape) / 2 + np.array(center_offset)
        # # x_i, y_i = np.where(im > flux_threshold)
        # y_i, x_i = np.where(im > flux_threshold)

        # x = (x_i - x_cen) * plate_scale
        # y = (y_i - y_cen) * plate_scale
        # # weight = im[x_i, y_i]
        # weight = im[y_i, x_i]

        # i = oversample
        # oset = np.linspace(-0.5, 0.5, 2*i+1)[1:2*i:2] * plate_scale

        # x_list, y_list, w_list = [], [], []
        # for i in oset:
            # for j in oset:
                # x_list += (x + i).tolist()
                # y_list += (y + j).tolist()
                # w_list += (weight / oversample**2).tolist()
        # x, y, weight = np.array(x_list), np.array(y_list), np.array(w_list)

        if oversample != 1:
            img = spi.zoom(images, oversample, order=3).astype(np.float32)
            scale_factor = np.sum(images)/np.sum(img)
            if conserve_flux:
                img *= scale_factor
        else:
            img = images
            scale_factor = 1

        # Ugly stripes are fixed - KL - 22.08.2017
        y_cen, x_cen = np.array(img.shape) // 2 + 0.5
        #y_cen, x_cen = np.array(img.shape) / 2
        y_i, x_i = np.where(img > flux_threshold * scale_factor)

        pix_res = plate_scale / oversample
        x = (x_i - x_cen) * pix_res + center_offset[0]
        y = (y_i - y_cen) * pix_res + center_offset[1]

        weight = img[y_i, x_i]
        ref = np.zeros(len(x))

        src = Source(lam=lam, spectra=spectra, x=x, y=y, ref=ref, weight=weight,
                     units=units, **kwargs)

        return src



def scale_spectrum(lam, spec, mag, filter_name="Ks", return_ec=False):
    """
    Scale a spectrum to be a certain magnitude

    Parameters
    ----------
    lam : np.ndarray
        [um] The wavelength bins for spectrum
    spec : np.ndarray
        The spectrum to be scaled into [ph/s/m2] for the given broadband filter
    mag : float
        magnitude of the source
    filter_name : str, TransmissionCurve, optional
           Any filter name from SimMETIS or a
           :class:`~.simmetis.spectral.TransmissionCurve` object
           (see :func:`~.simmetis.optics.get_filter_set`)
    return_ec : bool, optional
        If True, a :class:`simmetis.spectral.EmissionCurve` object is returned.
        Default is False

    Returns
    -------
    lam : np.ndarray
        [um] The centres of the wavelength bins for the new spectrum
    spec : np.ndarray
        [ph/s/m2] The spectrum scaled to the specified magnitude

    If return_ec == True, a :class:`simmetis.spectral.EmissionCurve` is returned

    See Also
    --------
    :class:`simmetis.spectral.TransmissionCurve`,
    :func:`simmetis.optics.get_filter_curve`,
    :func:`simmetis.optics.get_filter_set`,
    :func:`simmetis.source.SED`,
    :func:`simmetis.source.stars`

    Examples
    --------

    Scale the spectrum of a G2V star to J=25:

        >>> lam, spec = simmetis.source.SED("G2V")
        >>> lam, spec = simmetis.source.scale_spectrum(lam, spec, 25, "J")

    Scale the spectra for many stars to different H-band magnitudes:

        >>> from simmetis.source import SED, scale_spectrum
        >>>
        >>> star_list = ["A0V", "G2V", "M5V", "B6III", "O9I", "M2IV"]
        >>> magnitudes = [ 20,  25.5,  29.1,      17,  14.3,   22   ]
        >>> lam, spec = SED(star_list)
        >>> lam, spec = scale_spectrum(lam, spec, magnitudes, "H")

    Re-scale the above spectra to the same magnitudes in Pa-Beta

        >>> # Find which filters are in the simmetis/data directory
        >>>
        >>> import simmetis.optics as sim_op
        >>> print(sim_op.get_filter_set()       )
        ['B', 'BrGamma', 'CH4_169', 'CH4_227', 'FeII_166', 'H', 'H2O_204',
            'H2_212', 'Hcont_158', 'I', 'J', 'K', 'Ks', 'NH3_153', 'PaBeta',
            'R', 'U', 'V', 'Y', 'z']
        >>>
        >>> lam, spec = scale_spectrum(lam, spec, magnitudes, "PaBeta")


    """
    # The following was part of docstring. The example does not work, because
    # the new filter is not calibrated.
    #
    # Create a tophat filter and rescale to magnitudes in that band:
    #
    #     >>> # first make a tranmsission curve for the filter
    #     >>>
    #     >>> from simmetis.spectral import TransmissionCurve
    #     >>> filt_lam   = np.array([0.3, 1.09, 1.1, 1.15, 1.16, 3.])
    #     >>> filt_trans = np.array([0.,  0.,   1.,  1.,   0.,   0.])
    #     >>> new_filt   = TransmissionCurve(lam=filt_lam, val=filt_trans)
    #     >>>
    #     >>> lam, spec = scale_spectrum(lam, spec, magnitudes, new_filt)

    from simmetis.optics import get_filter_curve

    mag = np.asarray(mag)

    # Number of photons corresponding to desired apparent magnitude mag
    ideal_phs = zero_magnitude_photon_flux(filter_name) * 10**(-0.4 * mag)
    if isinstance(ideal_phs, (int, float)):
        ideal_phs = [ideal_phs]

    if len(spec.shape) == 1:
        spec = [spec]

    # Convert spectra to EmissionCurves
    curves = [EmissionCurve(lam=lam, val=sp, area=1, units="ph/s/m2")
              for sp in spec]

    if isinstance(filter_name, TransmissionCurve):
        filt = filter_name
    elif os.path.exists(filter_name):
        filt = TransmissionCurve(filename=filter_name)
    else:
        filt = get_filter_curve(filter_name)

    # Rescale the spectra
    for i in range(len(curves)):
        tmp = curves[i] * filt
        obs_ph = tmp.photons_in_range()
        scale_factor = ideal_phs[i] / obs_ph
        curves[i] *= scale_factor

    # Return in desired format
    if return_ec:
        if len(curves) > 1:
            return curves
        else:
            return curves[0]
    else:
        if len(curves) > 1:
            return curves[0].lam, np.array([curve.val for curve in curves])
        else:
            return curves[0].lam, curves[0].val


def scale_spectrum_sb(lam, spec, mag_per_arcsec, pix_res=0.004,
                      filter_name="Ks", return_ec=False):
    """
    Scale a spectrum to be a certain magnitude per arcsec2

    Parameters
    ----------
    lam : np.ndarray
        [um] The wavelength bins for spectrum
    spec : np.ndarray
        The spectrum to be scaled into [ph/s/m2] for the given broadband filter
    mag_per_arcsec : float
        [mag/arcsec2] surface brightness of the source
    pix_res : float
        [arcsec] the pixel resolution
    filter_name : str, TransmissionCurve
        Any filter name from SimMETIS or a
        :class:`~.simmetis.spectral.TransmissionCurve` object
        (see :func:`~.simmetis.optics.get_filter_set`)
    return_ec : bool, optional
        If True, a :class:`simmetis.spectral.EmissionCurve` object is returned.
        Default is False

    Returns
    -------
    lam : np.ndarray
        [um] The centres of the wavelength bins for the new spectrum
    spec : np.array
        [ph/s/m2/pixel] The spectrum scaled to the specified magnitude

    See Also
    --------

    """

    curve = scale_spectrum(lam, spec, mag_per_arcsec, filter_name,
                           return_ec=True)
    curve.val *= pix_res**2
    curve.params["pix_res"] = pix_res

    if return_ec:
        return curve
    else:
        return curve.lam, curve.val


def flat_spectrum(mag, filter_name="Ks", return_ec=False):
    """
    Return a flat spectrum scaled to a certain magnitude

    Parameters
    ----------
    mag : float
        [mag] magnitude of the source
    filter_name : str, TransmissionCurve, optional
        str - filter name. See ``simmetis.optics.get_filter_set()``. Default: "Ks"
        TransmissionCurve - output of ``simmetis.optics.get_filter_curve()``
    return_ec : bool, optional
        If True, a simmetis.spectral.EmissionCurve object is returned.
        Default is False

    Returns
    -------
    lam : np.ndarray
        [um] The centres of the wavelength bins for the new spectrum
    spec : np.array
        [ph/s/m2/arcsec] The spectrum scaled to the specified magnitude

    """
    lam = np.arange(3.0, 13.2, 0.01)
    spec = np.ones(len(lam))

    if return_ec:     # TODO: mag_per_arcsec undefined? (OC)
        curve = scale_spectrum(lam, spec, mag, filter_name,
                               return_ec)
        return curve
    else:
        lam, spec = scale_spectrum(lam, spec, mag, filter_name,
                                   return_ec)
        return lam, spec


def flat_spectrum_sb(mag_per_arcsec, filter_name="Ks", pix_res=0.004,
                     return_ec=False):
    """
    Return a flat spectrum for a certain magnitude per arcsec

    Parameters
    ----------
    mag_per_arcsec : float
        [mag/arcsec2] surface brightness of the source
    filter_name : str, TransmissionCurve, optional
        str - filter name. See ``simmetis.optics.get_filter_set()``. Default: "Ks"
        TransmissionCurve - output of ``simmetis.optics.get_filter_curve()``
    pix_res : float
        [arcsec] the pixel resolution. Default is 4mas (i.e. 0.004)
    return_ec : bool, optional
        Default is False. If True, a simmetis.spectral.EmissionCurve object is
        returned.

    Returns
    -------
    lam : np.ndarray
        [um] The centres of the wavelength bins for the new spectrum
    spec : np.array
        [ph/s/m2/arcsec] The spectrum scaled to the specified magnitude

    """
    lam = np.arange(3.0, 13.2, 0.01)
    spec = np.ones(len(lam))

    if return_ec:
        curve = scale_spectrum_sb(lam, spec, mag_per_arcsec, pix_res,
                                  filter_name, return_ec)
        return curve
    else:
        lam, spec = scale_spectrum_sb(lam, spec, mag_per_arcsec, pix_res,
                                      filter_name, return_ec)
        return lam, spec


def _rebin(img, bpix):
    '''Rebin image img by block averaging bpix x bpix pixels'''

    xedge = np.shape(img)[0] % bpix
    yedge = np.shape(img)[1] % bpix
    img_block = img[xedge:, yedge:]

    binim = np.reshape(img_block,
                       (int(img_block.shape[0]/bpix), bpix,
                        int(img_block.shape[1]/bpix), bpix))
    binim = np.mean(binim, axis=3)
    binim = np.mean(binim, axis=1)
    return binim


def get_lum_class_params(lum_class="V", cat=None):
    """
    Returns a table with parameters for a certain luminosity class

    Parameters
    ----------
    lum_class : str, optional
        Default is the main sequence ("V")

    Returns : astropy.Table object

    """
    import astropy.table as tbl

    if cat is None:
        cat = ioascii.read(__pkg_dir__+"/data/EC_all_stars.csv")

    t = []
    for row in cat:
        spt = row["Stellar_Type"]
        if spt[0] in "OBAFGKM" and \
           spt[-len(lum_class):] == lum_class and \
           len(spt) == 2 + len(lum_class):
            t += [row.data]

    t = tbl.Table(data=np.array(t), names=cat.colnames)

    return t


def get_nearest_spec_type(value, param="B-V", cat=None):
    """
    Return the spectral type of the star with the closest parameter value

    Compares values given for a certain stellar parameter and returns the spectral type
    which matches the best. In case several spectral types have the same value, only the first
    sectral type is returned

    Acceptable parameters are:
    "Mass" : [Msun]
    "Luminosity" : [Lsun]
    "Radius" : [Rsun]
    "Temp" : [K]
    "B-V" : [mag]
    "Mv" : [mag]
    "BC(Temp)" : [Corr]
    "Mbol" : [mag]

    Parameters
    ----------
    value : float, array
        The value that the spectral type should have
    param : str, optional
        Default is "B-V". The column to be searched.
    cat : astropy.Table, optional
        The catalogue to use. Default is in the simmetis/data directory

    Returns
    -------
    a value/list of strings corresponding to the spectral types which best fit to the given values

    """

    if cat is None:
        cat = ioascii.read(__pkg_dir__+"/data/EC_all_stars.csv")

    if isinstance(value, (np.ndarray, list, tuple)):
        spt = []
        for val in value:
            spt += [get_nearest_spec_type(val, param, cat)]

        return spt

    col = cat[param]
    i = np.argmin(np.abs(col-value))
    spec_type = cat["Stellar_Type"][i]

    return spec_type


def spectrum_sum_over_range(lam, flux, lam_min=None, lam_max=None):
    """Sum spectrum over range lam_min to lam_max

    Parameters
    ----------
    lam : float, array
        wavelength array of spectrum
    flux : float, array
        flux array of spectrum [ph/s/m2/bin]
    lam_min, lam_max : float
        wavelength limits of range over which the spectrum is summed. If None,
        the spectrum is summed over its definition range

    Returns
    -------
    number of photons within lam_min and lam_max [ph/s/m2]
    """
    if lam_min is None:
        lam_min = lam[0]
    if lam_max is None:
        lam_max = lam[-1]

    if lam_max < lam_min:
        raise ValueError("lam_max < lam_min")

    # Check if the slice limits are within the spectrum wavelength range
    dlam = lam[1] - lam[0]
    if (lam_min > lam[-1] + dlam/2) or (lam_max < lam[0] - dlam/2):
        print((lam_min, lam_max), (lam[0], lam[-1]))
        warnings.warn("lam_min or lam_max outside wavelength range" +
                      " of spectra. Returning 0 photons for this range")
        return np.array([0])

    # find the closest indices imin, imax that match the limits
    imin = np.argmin(np.abs(lam - lam_min))
    imax = np.argmin(np.abs(lam - lam_max))

    # Treat edge bins: Since lam[imin] < lam_min < lam_max < lam[imax], we have to
    # subtract part of the outer bins
    dlam = lam[1] - lam[0]
    spec_photons = np.sum(flux[:, imin:(imax + 1)], axis=1) \
                   - flux[:, imin] * (0.5 + (lam_min - lam[imin])/dlam) \
                   - flux[:, imax] * (0.5 - (lam_max - lam[imax])/dlam)

    return spec_photons


def load(filename):
    '''Load :class:'Source' object from filename'''
    return Source.load(filename)



###############################################################


"""
A bunch of helper functions to generate galaxies in SimMETIS
"""


def sie_grad(x, y, par):
    """
    Compute the deflection of an SIE (singular isothermal ellipsoid) potential

    Parameters
    ----------
    x, y : meshgrid arrays
        vectors or images of coordinates; should be matching numpy ndarrays

    par : list
        vector of parameters with 1 to 5 elements, defined as follows:
        par[0]: lens strength, or 'Einstein radius'
        par[1]: (optional) x-center (default = 0.0)
        par[2]: (optional) y-center (default = 0.0)
        par[3]: (optional) axis ratio (default=1.0)
        par[4]: (optional) major axis Position Angle
                in degrees c.c.w. of x axis. (default = 0.0)


    Returns
    -------
    xg, yg : gradients at the positions (x, y)


    Notes
    -----
    This routine implements an 'intermediate-axis' convention.
      Analytic forms for the SIE potential can be found in:
        Kassiola & Kovner 1993, ApJ, 417, 450
        Kormann et al. 1994, A&A, 284, 285
        Keeton & Kochanek 1998, ApJ, 495, 157
      The parameter-order convention in this routine differs from that
      of a previous IDL routine of the same name by ASB.


    Credit
    ------
    Adam S. Bolton, U of Utah, 2009

    http://www.physics.utah.edu/~bolton/python_lens_demo/

    """
    # Set parameters:
    b = np.abs(par[0]) # can't be negative!!!
    xzero = 0. if (len(par) < 2) else par[1]
    yzero = 0. if (len(par) < 3) else par[2]
    q = 1. if (len(par) < 4) else np.abs(par[3])
    phiq = 0. if (len(par) < 5) else par[4]
    eps = 0.001 # for sqrt(1/q - q) < eps, a limit expression is used.

    # Handle q > 1 gracefully:
    if (q > 1.):
        q = 1.0 / q
        phiq = phiq + 90.0

    # Go into shifted coordinats of the potential:
    phirad = np.deg2rad(phiq)
    xsie = (x-xzero) * np.cos(phirad) + (y-yzero) * np.sin(phirad)
    ysie = (y-yzero) * np.cos(phirad) - (x-xzero) * np.sin(phirad)

    # Compute potential gradient in the transformed system:
    r_ell = np.sqrt(q * xsie**2 + ysie**2 / q)
    qfact = np.sqrt(1./q - q)

    # (r_ell == 0) terms prevent divide-by-zero problems
    if (qfact >= eps):
        xtg = (b/qfact) * np.arctan(qfact * xsie / (r_ell + (r_ell == 0)))
        ytg = (b/qfact) * np.arctanh(qfact * ysie / (r_ell + (r_ell == 0)))
    else:
        xtg = b * xsie / (r_ell + (r_ell == 0))
        ytg = b * ysie / (r_ell + (r_ell == 0))

    # Transform back to un-rotated system:
    xg = xtg * np.cos(phirad) - ytg * np.sin(phirad)
    yg = ytg * np.cos(phirad) + xtg * np.sin(phirad)

    # Return value:
    return (xg, yg)



def apply_grav_lens(image, x_cen=0, y_cen=0, r_einstein=None, eccentricity=1,
                    rotation=0):
    """
    Apply a singular isothermal ellipsoid (SIE) gravitational lens to an image

    Parameters
    ----------
    image : np.ndarray

    x_cen, y_cen : float
        [pixel] centre of the background image relative to the centre of the
        field of view

    r_einstein : float
        [pixel] Einstein radius of lens.
        If None, r_einstein = image.shape[0] // 4

    eccentricity : float
        [1..0] The ratio of semi-minor to semi-major axis for the lens

    rotation : float
        [degrees] Rotation of lens ccw from the x axis


    Returns
    -------
    lensed_image : np.ndarray


    Example
    -------

        >>> from astropy.io import fits
        >>> im = fits.getdata("my_galaxy.fits")
        >>> im2 = apply_grav_lens(im, x_cen=30, rotation=-45, eccentricity=0.5,
                                  r_einstein=300)


    """

    if r_einstein is None:
        r_einstein = image.shape[0] // 4

    shifted_image = spi.shift(image, (x_cen, y_cen))

    nx, ny = shifted_image.shape
    w = np.linspace(-nx // 2, nx // 2, nx)
    h = np.linspace(-ny // 2, ny // 2, ny)
    x, y = np.meshgrid(w,h)

    # Get the distortions from the lens
    lpar = np.asarray([r_einstein, x_cen, y_cen, eccentricity, rotation])
    xg, yg = sie_grad(x, y, lpar)

    # Pull out the pixels from the original image and place them where the lens
    #  would put them
    i = (x-xg + nx//2).astype(int)
    j = (y-yg + ny//2).astype(int)

    lensed_image = shifted_image[j.flatten(),
                                 i.flatten()].reshape(shifted_image.shape)

    return lensed_image



def elliptical(half_light_radius, plate_scale, magnitude=10, n=4,
           filter_name="Ks", normalization="total", spectrum="elliptical",
           **kwargs):
    """
    Create a extended :class:`.Source` object for a "Galaxy"

    Parameters
    ----------
    half_light_radius : float
        [arcsec]

    plate_scale : float
        [arcsec]

    magnitude : float
        [mag, mag/arcsec2]

    n : float, optional
        Power law index. Default = 4
        - n=1 for exponential (spiral),
        - n=4 for de Vaucouleurs (elliptical)

    filter_name : str, TransmissionCurve, optional
        Default is "Ks". Values can be either:
        - the name of a SimMETIS filter : see optics.get_filter_set()
        - or a TransmissionCurve containing a user-defined filter

    normalization : str, optional
        ["half-light", "centre", "total"] Where the profile equals unity
        If normalization equals:
        - "half-light" : the pixels at the half-light radius have a surface
                         brightness of ``magnitude`` [mag/arcsec2]
        - "centre" : the maximum pixels have a surface brightness of
                     ``magnitude`` [mag/arcsec2]
        - "total" : the whole image has a brightness of ``magnitude`` [mag]

    spectrum : str, EmissionCurve, optional
        The spectrum to be associated with the galaxy. Values can either be:
        - the name of a SimMETIS SED spectrum : see get_SED_names()
        - an EmissionCurve with a user defined spectrum


    Optional Parameters (passed to ``sersic_profile``)
    --------------------------------------------------
    ellipticity : float
        Default = 0.5

    angle : float
        [deg] Default = 30. Rotation anti-clockwise from the x-axis

    width, height : int
        [arcsec] Dimensions of the image. Default: 512*plate_scale

    x_offset, y_offset : float
        [arcsec] The distance between the centre of the profile and the centre
        of the image. Default: (dx,dy) = (0,0)


    Returns
    -------
    galaxy_src : simmetis.Source


    See Also
    --------
    source.sersic_profile()
    optics.get_filter_set(), source.get_SED_names()
    spectral.TransmissionCurve, spectral.EmissionCurve


    """

    params = {"n"           : n,
              "ellipticity" : 0.5,
              "angle"       : 30,
              "width"       : plate_scale * 512,
              "height"      : plate_scale * 512,
              "x_offset"    : 0,
              "y_offset"    : 0}
    params.update(kwargs)

    pixular_hlr = half_light_radius / plate_scale

    im = sersic_profile(r_eff        =pixular_hlr,
                        n            =params["n"],
                        ellipticity  =params["ellipticity"],
                        angle        =params["angle"],
                        normalization=normalization,
                        width        =params["width"] /plate_scale,
                        height       =params["height"]/plate_scale,
                        x_offset     =params["x_offset"]/plate_scale,
                        y_offset     =params["y_offset"]/plate_scale)

    if isinstance(spectrum, EmissionCurve):
        lam, spec = spectrum.lam, spectrum.val
        lam, spec = scale_spectrum(lam=lam, spec=spec, mag=magnitude,
                                   filter_name=filter_name)
    elif spectrum in get_SED_names():
        lam, spec = SED(spec_type=spectrum, filter_name=filter_name,
                        magnitude=magnitude)
    else:
        print(spectrum)
        raise ValueError("Cannot understand ``spectrum``")

    galaxy_src = source_from_image(images=im, lam=lam, spectra=spec,
                                   plate_scale=plate_scale)

    return galaxy_src


def sersic_profile(r_eff=100, n=4, ellipticity=0.5, angle=30,
                   normalization="total",
                   width=1024, height=1024, x_offset=0, y_offset=0,
                   oversample=1):
    """
    Returns a 2D array with a normalised Sersic profile

    Parameters
    ----------
    r_eff : float
        [pixel] Effective (half-light) radius

    n : float
        Power law index.
        - n=1 for exponential (spiral),
        - n=4 for de Vaucouleurs (elliptical)

    ellipticity : float
        Ellipticity is defined as (a - b)/a. Default = 0.5

    angle : float
        [deg] Default = 30. Rotation anti-clockwise from the x-axis

    normalization : str, optional
        ["half-light", "centre", "total"] Where the profile equals unity
        If normalization equals:
        - "half-light" : the pixels at the half-light radius are set to 1
        - "centre" : the maximum values are set to 1
        - "total" : the image sums to 1

    width, height : int
        [pixel] Dimensions of the image

    x_offset, y_offset : float
        [pixel] The distance between the centre of the profile and the centre
        of the image

    oversample : int
        Factor of oversampling, default factor = 1. If > 1, the model is
        discretized by taking the average of an oversampled grid.

    Returns
    -------
    img : 2D array


    Notes
    -----
    Most units are in [pixel] in this function. This differs from
    :func:`.galaxy` where parameter units are in [arcsec] or [pc]

    """

    from astropy.modeling.models import Sersic2D

    # Silently cast to integer
    os_factor = np.int(oversample)

    if os_factor <= 0:
        raise ValueError("Oversampling factor must be >=1.")

    width_os = os_factor * width
    height_os = os_factor * height
    x, y = np.meshgrid(np.arange(width_os), np.arange(height_os))

    dx = 0.5 * width_os  + x_offset * os_factor
    dy = 0.5 * height_os + y_offset * os_factor

    r_eff_os = r_eff * os_factor

    mod = Sersic2D(amplitude=1, r_eff=r_eff_os, n=n, x_0=dx, y_0=dy,
                   ellip=ellipticity, theta=np.deg2rad(angle))
    img_os = mod(x, y)

    # Rebin os_factord image
    img = _rebin(img_os, os_factor)

    thresh = np.max([img[0,:].max(), img[-1,:].max(),
                     img[:,0].max(), img[:,-1].max()])
    img[img < thresh] = 0

    if "cen" in normalization.lower():
        img /= np.max(img)
    elif "tot" in normalization.lower():
        img /= np.sum(img)

    return img


def spiral_profile(r_eff, ellipticity=0.5, angle=45,
                   n_arms=2, tightness=4., arms_width=0.1, central_brightness=10,
                   normalization='total', width=1024, height=1024, oversample=1,
                   **kwargs):
    """
    Creates a spiral profile with arbitary parameters

    Parameters
    ----------
     r_eff : float
        [pixel] Effective (half-light) radius

    ellipticity : float
        Ellipticity is defined as (a - b)/a. Default = 0.5

    angle : float
        [deg] Default = 45. Rotation anti-clockwise from the x-axis

    n_arms : int
        Number of spiral arms

    tightness : float
        How many times an arm crosses the major axis. Default = 4.

    arms_width : float
        An arbitary scaling factor for how think the arms should be.
        Seems to scale with central_brightness. Default = 0.1

    central_brightness : float
        An arbitary scaling factor for the strength of the central region.
        Has some connection to ars_width. Default = 10

    normalization : str, optional
        ["half-light", "centre", "total"] Where the profile equals unity
        If normalization equals:
        - "centre" : the maximum values are set to 1
        - "total" : the image sums to 1

    width, height : int, int
        [pixel] Dimensions of the image

    x_offset, y_offset : float
        [pixel] The distance between the centre of the profile and the centre
        of the image

    oversample : int
        Factor of oversampling, default factor = 1. If > 1, the model is
        discretized by taking the average of an oversampled grid.


    Optional Parameters
    -------------------
    **kwargs are passed to sersic_profile()


    Returns
    -------
    img : np.ndarray
        A 2D image of a spiral disk


    Notes
    -----
    The intensity drop-off is dictated by a sersic profile of with indes n=1,
    i.e. an exponential drop-off. This can be altered by passing the keyword
    "n=" as an optional parameter.

    Spiral structure taken from here:
    https://stackoverflow.com/questions/36095775/creating-a-spiral-structure-in-python-using-hyperbolic-tangent


    See Also
    --------
    sersic_profile()


    """

    if ellipticity >= 1.:
        raise ValueError("ellipticiy <= 1 . This is physically meaningless")

    # create a spiral
    xx, yy = np.meshgrid(np.arange(-width/2, width/2),
                         np.arange(-height/2, height/2))
    r = np.sqrt(abs(xx)**2 + abs(yy)**2)

    spiral = np.cos( n_arms * np.arctan2(xx,yy) + tightness * np.log(r**2) ) / \
             arms_width + central_brightness

    spiral[spiral < 0] = 0

    # add an exponential drop off in light intensity for the disk
    disk = sersic_profile(r_eff=r_eff, n=1, ellipticity=0, angle=0,
                          normalization=normalization, oversample=oversample,
                          width=width, height=height, **kwargs)

    img = spiral * disk
    thresh = np.max([img[0,:].max(), img[-1,:].max(),
                     img[:,0].max(), img[:,-1].max()])
    img[img < thresh] = 0

    # rotate and tilt
    ab = 1 - ellipticity
    img= spi.zoom(img, (ab, 1), order=1)
    img = spi.rotate(img, angle, order=1)

    # normalise the flux
    img[img < 0] = 0
    img = np.nan_to_num(img)
    if "cen" in normalization.lower():
        img /= np.max(img)
    elif "tot" in normalization.lower():
        img /= np.sum(img)

    return img



def spiral(half_light_radius, plate_scale, magnitude=10,
           filter_name="Ks", normalization="total", spectrum="spiral",
           **kwargs):
    """
    Create a extended :class:`.Source` object for a "Galaxy"

    Parameters
    ----------
    half_light_radius : float
        [arcsec]

    plate_scale : float
        [arcsec]

    magnitude : float
        [mag, mag/arcsec2]

    filter_name : str, TransmissionCurve, optional
        Default is "Ks". Values can be either:
        - the name of a SimMETIS filter : see optics.get_filter_set()
        - or a TransmissionCurve containing a user-defined filter

    normalization : str, optional
        ["half-light", "centre", "total"] Where in the profile equals unityy
        If normalization equals:
        - "half-light" : the pixels at the half-light radius have a surface
                         brightness of ``magnitude`` [mag/arcsec2]
        - "centre" : the maximum pixels have a surface brightness of
                     ``magnitude`` [mag/arcsec2]
        - "total" : the whole image has a brightness of ``magnitude`` [mag]

    spectrum : str, EmissionCurve, optional
        The spectrum to be associated with the galaxy. Values can either be:
        - the name of a SimMETIS SED spectrum : see get_SED_names()
        - an EmissionCurve with a user defined spectrum


    Optional Parameters (passed to ``spiral_profile``)
    --------------------------------------------------
    n_arms : int
        Number of spiral arms

    tightness : float
        How many times an arm crosses the major axis. Default = 4.

    arms_width : float
        An arbitary scaling factor for how think the arms should be.
        Seems to scale with central_brightness. Default = 0.1

    central_brightness : float
        An arbitary scaling factor for the strength of the central region.
        Has some connection to ars_width. Default = 10

    ellipticity : float
        Default = 0.5

    angle : float
        [deg] Default = 30. Rotation anti-clockwise from the x-axis

    n : float
         Sersic index, default = 1 (exponential disk)

    width, height : int
        [arcsec] Dimensions of the image. Default: 512*plate_scale


    Returns
    -------
    galaxy_src : simmetis.Source


    See Also
    --------
    sersic_profile(), spiral_profile()
    optics.get_filter_set(), source.get_SED_names()
    spectral.TransmissionCurve, spectral.EmissionCurve

    """

    pixular_hlr = half_light_radius / plate_scale

    params = {"n"           : 1,
              "ellipticity" : 0.5,
              "angle"       : 30,
              "width"       : pixular_hlr,
              "height"      : pixular_hlr,
              "n_arms"      : 2,
              "tightness"   : 4.,
              "arms_width"  : 0.1,
              "central_brightness" : 10}
    params.update(kwargs)


    spiral = spiral_profile(r_eff             =pixular_hlr,
                            ellipticity       =params["ellipticity"],
                            angle             =-params["angle"],
                            normalization     =normalization,
                            #width             =params["width"],
                            #height            =params["height"],
                            width             =2*pixular_hlr,
                            height            =2*pixular_hlr,
                            n_arms            =params["n_arms"],
                            tightness         =params["tightness"],
                            arms_width        =params["arms_width"],
                            central_brightness=params["central_brightness"])

    disk = sersic_profile(r_eff        =pixular_hlr,
                          n            =1,
                          ellipticity  =params["ellipticity"],
                          angle        =params["angle"],
                          normalization=normalization,
                          width        =spiral.shape[1],
                          height       =spiral.shape[0])

    thresh = np.max((disk[0,:].max(), disk[-1,:].max(),
                     disk[:,0].max(), disk[:,-1].max()))
    disk[disk < thresh] = 0


    if isinstance(spectrum, EmissionCurve):
        lam, spec = spectrum.lam, spectrum.val
        lam, spec = scale_spectrum(lam=lam, spec=spec, mag=magnitude,
                                   filter_name=filter_name)
    elif spectrum in get_SED_names():
        lam, spec = SED(spec_type=spectrum, filter_name=filter_name,
                        magnitude=magnitude)
    else:
        print(spectrum)
        raise ValueError("Cannot understand ``spectrum``")

    gal_img = (spiral + disk).T

    galaxy_src = source_from_image(images=gal_img, lam=lam, spectra=spec,
                                   plate_scale=plate_scale)

    return galaxy_src



def _rebin(img, bpix):
    '''Rebin image img by block averaging bpix x bpix pixels'''

    xedge = np.shape(img)[0] % bpix
    yedge = np.shape(img)[1] % bpix
    img_block = img[xedge:, yedge:]

    binim = np.reshape(img_block,
                       (int(img_block.shape[0]/bpix), bpix,
                        int(img_block.shape[1]/bpix), bpix))
    binim = np.mean(binim, axis=3)
    binim = np.mean(binim, axis=1)
    return binim
