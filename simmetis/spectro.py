#!/usr/bin/env python3
# coding=latin1
"""
spectro.py
Created:     Sat Oct 27 14:52:39 2018 by Koehler@Quorra
Last change: Wed Nov 14 15:51:35 2018

Python-script to simulate LMS of METIS

TODO: easy switch for best/median/poor conditions
"""

from datetime import datetime

import numpy as np

from scipy.interpolate import interp1d, RectBivariateSpline

from astropy import units as u
from astropy.io import fits
from astropy import constants as const
from astropy import wcs
import astropy.convolution as ac

import matplotlib.pyplot as plt

import simmetis as sm
import simmetis.spectral as sc

#############################################################################

def _scale_image(img, scale):
    '''interpolate image to new shape'''

    naxis2, naxis1 = img.shape
    print("scale_image: old shape =", img.shape)

    in_x = np.arange(naxis1)
    in_y = np.arange(naxis2)
    #print(len(in_x))

    # TODO better: scale from the center of the img

    # we need to scale the coord of the last pixel, not the pixel behind the end!
    out_x = np.arange(round((naxis1-1)*scale)+1) / scale
    out_y = np.arange(round((naxis2-1)*scale)+1) / scale
    #print(len(out_x))

    interp = RectBivariateSpline(in_x, in_y, img, kx=1, ky=1)	# bilinear interpol
    scaled_img = interp(out_x, out_y, grid=True)

    print("scale_image: new shape =", scaled_img.shape)

    return scaled_img


#############################################################################

class LMS:
    """
    Simulate the LM-spectrograph

    Parameters:
    filename: Name of input data cube
    lambda0: wavelength at zero-point of 3.dimension
    """

    def __init__(self, filename, config, lambda0=3.8):
        '''Read datacube and WCS'''

	# separate our output from the bullshit printed by import
        print('============================================')

        self.filename = filename
        print("Source file ", filename)

        self.src_fits = fits.open(filename)
        self.src_cube = self.src_fits[0].data
        self.src_header = self.src_fits[0].header

        naxis3, naxis2, naxis1 = self.src_cube.shape
        print(naxis1, "x", naxis2, "pixels,", naxis3, "spectral channels")

        # Parse the WCS keywords in primary HDU
        self.wcs = wcs.WCS(self.src_header)

        pixscale1, pixscale2 = wcs.utils.proj_plane_pixel_scales(self.wcs)[0:2]
        pixscale1 = (pixscale1 * self.wcs.wcs.cunit[0]).to(u.mas)
        pixscale2 = (pixscale2 * self.wcs.wcs.cunit[1]).to(u.mas)
        #print("image pixscale from WCS:", pixscale1, pixscale2, "/pixel")

        self.src_pixscale = np.asarray((pixscale1.value, pixscale2.value))*u.mas
        print("image pixscale:", self.src_pixscale, "per pixel")

        # Check flux units of data cube

        try:
            bunit = self.src_header['BUNIT']
            print("BUNIT:", bunit)
            if bunit.lower() == 'jy/pixel':
                #plt.semilogy(self.src_cube[:,111,100])
                #print(np.max(self.src_cube[:,111,100]))
                pixarea = (wcs.utils.proj_plane_pixel_area(self.wcs) *
                           self.wcs.wcs.cunit[0]*self.wcs.wcs.cunit[1]).to(u.arcsec*u.arcsec)
                print("Pixel area is", pixarea)
                self.src_cube /= pixarea.value
                #plt.semilogy(self.src_cube[:,111,100])
                #plt.show()
                #print(np.max(self.src_cube[:,111,100]))
        except KeyError:
            print("Keyword BUNIT not found.  Assuming the data is in units of Jy/arcsec2")

        # make a Nlambda x 3 array
        crd = np.zeros((naxis3, 3))
        crd[:,2] = np.arange(naxis3)

        # Convert pixel coordinates to world coordinates
        # Second argument is "origin" -- in this case 0-based coordinates
        world_coo = self.wcs.wcs_pix2world(crd, 0)[:,2] * self.wcs.wcs.cunit[2]

        print("CTYPES:", self.wcs.wcs.ctype)

        if self.wcs.wcs.ctype[2] == 'VELO':
            #print("Velocities:",world_coo)
            # this should be in the header, shouldn't it?
            restcoo = lambda0 * u.um			   # unit of lambda0 must be micron
            wavelen = restcoo * (1. + world_coo / const.c)
            #print("Wavelengths:",wavelen)
        elif self.wcs.wcs.ctype[2] == 'WAVE':
            # TODO: test me!
            wavelen = world_coo.to(u.um)
            restcoo = self.wcs.wcs.restwav * self.wcs.wcs.cunit[2]
        elif self.wcs.wcs.ctype[2] == 'FREQ':
            wavelen = world_coo.to(u.um, equivalencies=u.spectral())
            restcoo = self.wcs.wcs.restfrq * self.wcs.wcs.cunit[2]
        else:
            raise NotImplementedError('spectral axis must have type VELO, WAVE, or FREQ')

        print("Wavelengths:", wavelen[0:3], "...", wavelen[-1])
        self.wavelen = wavelen.value	# should we store it with units?
        self.restcoo = restcoo		# RESTFRQ or RESTWAV with units

        print("restfrq:", self.wcs.wcs.restfrq)
        print("restwav:", self.wcs.wcs.restwav)
        #print("Rest wavelen:", restcoo.to(u.um, equivalencies=u.spectral()))

        in_velos = wavelen.to(u.m/u.s, equivalencies=u.doppler_optical(restcoo))
        print("Source velocities (WCS):", in_velos[0], "...", in_velos[-1])
        #print(in_velos)

        step = 1.5 * u.km/u.s

        new3 = int((in_velos[-1] - in_velos[0]) / step)+1
        print("new naxis3:", new3)

        meanv = (in_velos[0] + in_velos[-1]) / 2.
        print("middle v:", meanv)

        self.det_velocities = np.arange((meanv-step*(new3-1)/2.).to(u.m/u.s).value,
                                        (meanv+step*((new3-1)/2.+1)).to(u.m/u.s).value,
                                        step.to(u.m/u.s).value)
        print("Detector velocities:", self.det_velocities.shape,
              self.det_velocities[0], "...", self.det_velocities[-1])

        # initialize the target cube, in case someone does not call transmission_emission()

        self.target_cube = self.src_cube
        self.target_hdr = self.src_header.copy()
        self.target_hdr['BUNIT'] = 'Jy/arcsec2'
        self.background = None

        self.emission = None	# will be computed later

        # initialize SimCADO to set searchpaths
        #
        print("Reading config ", config)
        self.cmds = sm.UserCommands(config)

        self.det_pixscale = 8.2		# mas/pixel, not quite fixed yet


    #############################################################################

    def make_target_header(self):

        self.target_hdr['DATE'] = datetime.now().isoformat()

        for key in self.cmds.cmds:
            val = self.cmds.cmds[key]

            if isinstance(val, (sc.TransmissionCurve, sc.EmissionCurve,
                                sc.UnityCurve, sc.BlackbodyCurve)):
                val = val.params["filename"]

            if isinstance(val, str) and len(val) > 35:
                val = "... " + val[-35:]

            self.target_hdr["HIERARCH "+key] = val



    #############################################################################

    def transmission_emission(self, conditions='median', plot=False):
        '''
        Apply transission to src_cube and calculate emission spectrum
        '''

        print("-----Transmission and Emission-----")
        if plot:
            print("Plotting pixel [111,100] from source cube")
            plt.plot(self.wavelen, self.src_cube[:,111,100])
            plt.title("Pixel [111,100] in source cube")
            plt.show()

        # calculate wavelens of detector, needed for emission
        #
        det_wavelen = self.det_velocities * u.m/u.s
        det_wavelen = det_wavelen.to(u.um, equivalencies=u.doppler_optical(self.restcoo))

        #############################################################################
        # Read in trans-/emission from skycal
        # TODO: set config and/or telescope parameters, too

        if conditions == 'best':
            skyfile = 'skycal_R308296_best_conditions.fits'
        elif conditions == 'median':
            skyfile = 'skycal_R308296_median_conditions.fits'
        elif conditions == 'poor':
            skyfile = 'skycal_R308296_poor_conditions.fits'
        else:
            raise ValueError('Undefined conditions "'+conditions+
                             '", only "best", "median", and "poor" are defined')

        skyfile = sm.utils.find_file(skyfile)
        skytransfits = fits.open(skyfile)
        print("Reading ", skyfile)

        skytrans = skytransfits[1].data

        skylam = skytrans['LAMBDA']
        skytran = skytrans['TRANS']
        skyemis = skytrans['EMIS']

        #plt.plot(skylam, skytran)
        #plt.show()

        idx = (np.where((skylam >= np.min(self.wavelen)) &
                        (skylam <= np.max(self.wavelen))))[0]
        print("Index skytran to src-wave:", idx[0], "...", idx[-1])
        #
        # interpolate transmission/emission onto source-grid
        #
        transmission = np.interp(self.wavelen, skylam, skytran) # dimensionless, [0...1]
        emission     = np.interp(det_wavelen, skylam, skyemis)  # photons/s/um/m^2/arcsec^2

        if plot:
            print("Plotting sky transmission in source wavelength range...")
            plt.plot(skylam[idx], skytran[idx], "+")
            plt.plot(self.wavelen, transmission)
            plt.title("Transmission")
            #plt.show()

        #############################################################################
        # get trans-/emission from SimMETIS
        #
        # TODO: set best/median/poor conditions in optical train
        #
        print("Steaming up optical train")

        self.cmds["SIM_VERBOSE"] = 'hell no!'
        self.cmds["INST_FILTER_TC"] = "TC_filter_open.dat"
	# open filter, but remember that the detector cuts off at 5.5um

        self.cmds["SIM_DETECTOR_PIX_SCALE"] = 1.
        # We set the pixel size to 1 arcsec^2, this will give us emission per arcsec^2
        # TODO: use real pixel size in confg, get it from cmds instead?

        # Create transmission curve.
        opttrain = sm.OpticalTrain(self.cmds)

        # Some comments about the OpticalTrain in SimMumble:
        #
        # For MICADO, AO means MAORY
        # For METIS, this should be configured to do nothing
        #
        # tc_ao = 	 transmission of everything after first mirror in AO system
        #		 (n-1 AO mirrors, entrance window, dichroic, instrument...)
        # tc_mirror= transmission of everything after the telescope
        #		 (tc_ao + one AO mirror, but not M1)
        # tc_atmo =	 transmission of telescope and instrument
        #		 (i.e. _without_ the atmosphere)
        # tc_source= transmission seen by light from the source
        #		 (including atmosphere)
        #
        # ph_ao    = blackbody @ INST_AO_TEMPERATURE * 0.1 * tc_ao
        # ph_mirror= _gen_thermal_emission (based on SCOPE_MIRROR_LIST) * tc_mirror
        # ph_atmo  = atmospheric emission * tc_atmo
        #
        tc_lam = opttrain.tc_atmo.lam
        tc_val = opttrain.tc_atmo.val
        tc_trans = interp1d(tc_lam, tc_val, kind='linear', bounds_error=False, fill_value=0.)

        if plot:
            print("Plotting SimMetis-transmission in source wavelength range...")
            plt.plot(self.wavelen, tc_trans(self.wavelen))
            plt.show()

        transmission *= tc_trans(self.wavelen)

        #############################################################################
        #
        # transmit the source cube!
        #
        self.target_cube = self.src_cube * transmission[:, np.newaxis, np.newaxis]
        self.make_target_header()

        if plot:
            print("Plotting pixel [111,100] from transmitted cube")
            plt.plot(self.wavelen, self.target_cube[:,111,100])
            plt.title("Pixel [111,100] in transmitted cube")
            plt.show()

        #############################################################################
        # let there be emitted light!
        #
        # this is the code in spectral.py/BlackbodyCurve to compute the bin width
        lam = opttrain.ph_mirror.lam
        lam_res = lam[1] - lam[0]
        edges = np.append(lam - 0.5 * lam_res, lam[-1] + 0.5 * lam_res)
        lam_res = edges[1:] - edges[:-1]

        # lam_res is now (approximately) the bin width in um
        #
        # calculate emission in photons/s/um/pixel

        ph_mirr_um = opttrain.ph_mirror.val / lam_res

        if opttrain.ph_ao is not None:
            ph_mirr_um += opttrain.ph_ao.val / lam_res

        ph_mirror = interp1d(lam, ph_mirr_um,
                             kind='linear', bounds_error=False, fill_value=0.)

        print("Pix_res: ", opttrain.cmds.pix_res)

        mirr_list = self.cmds.mirrors_telescope
        mirr_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                       mirr_list["Inner"]**2)
        print("Mirror area: ", mirr_area, "[m^2]")

        ph_atmo_um = emission * mirr_area * tc_trans(det_wavelen)
        ph_mirrors = ph_mirror(det_wavelen)

        if plot:
            print("Plotting atmospheric & mirror emission...")
            plt.plot(det_wavelen, ph_atmo_um, '+')
            plt.plot(det_wavelen, ph_mirrors)
            plt.plot(det_wavelen, ph_atmo_um+ph_mirrors)
            plt.title("Emission")
            plt.show()

        self.emission = ph_atmo_um+ph_mirrors	# photons/s/um/arcsec2 - change to /pixel?


    #############################################################################

    def convolve_psf(self, psf_name, write_psf=None, plot=False):
        '''Convolve target_cube with Point Spread Function'''

        print("-----Convolution with PSF-----")

        print("image pixscale from WCS:", self.src_pixscale, "/ pixel")
        if self.src_pixscale[0] != self.src_pixscale[1]:
            print("pixels are not square!  Bad things will happen!")

        psf_fits = fits.open(psf_name)

        ext = psf_fits.index_of("PSF_3.80UM")
        print("PSF for 3.8um is extension", ext)
        psf_img = psf_fits[ext].data
        psf_hdr = psf_fits[ext].header

        psf_pixscale = psf_hdr['PIXSCALE']
        print("PSF pixel scale is",
              psf_pixscale, "mas/pix @ 3.8 um,", psf_img.shape, "pixels")

        psf_pixscale *= np.mean(self.wavelen) / 3.8
        print("PSF pixel scale is", psf_pixscale, "mas/pix @", np.mean(self.wavelen), "um,")

        scale = psf_pixscale / self.src_pixscale[0].value
        print("Scale factor", scale)

        psf_scaled = _scale_image(psf_img, scale)

        norm = np.sum(psf_scaled)
        print("Normalizing PSF by factor", norm)
        psf_scaled /= norm

        # crop to size needed for input data
        # parameters found empirically to center peak of PSF
        # (would be better if _scale_image would scale from the center)

        psf_center = np.asarray(psf_scaled.shape) // 2
        img_shape = self.target_cube.shape[1:3]
        print("Cropping PSF to [",
              psf_center[0]-img_shape[0], ":", psf_center[0]+img_shape[0]+1, ",",
              psf_center[1]-img_shape[1], ":", psf_center[1]+img_shape[1]+1, "]")

        psf_scaled = psf_scaled[(psf_center[0]-img_shape[0]):(psf_center[0]+img_shape[0]+1),
                                (psf_center[1]-img_shape[1]):(psf_center[1]+img_shape[1]+1)]

        if write_psf is not None:
            print("Writing scaled PSF to ", write_psf)
            psf_hdr['EXTNAME'] = None
            hdu = fits.PrimaryHDU(psf_scaled)
            hdu.header['WAVELENG'] = np.mean(self.wavelen)
            hdu.header['PIXSCALE'] = psf_pixscale
            hdu.writeto(write_psf, overwrite=True)

        #############################################################################
        print("Convolving with PSF...")

        for i in range(self.target_cube.shape[0]):
            print(self.target_cube.shape[0]-i, end=' ', flush=True)
            self.target_cube[i,:,:] = ac.convolve_fft(self.target_cube[i,:,:], psf_scaled)
            # Note: convolve_fft sets values outside the image bounds to 0.

        print()
        if plot:
            plt.plot(self.target_cube[:,111,100])
            plt.title("Pixel [111,100] after convolution with PSF")


    #############################################################################
    def convolve_lsf(self, plot=False):
        '''Convolve with Line Spread Function'''

        # TODO: interpolate to velocity-grid before convolving
        #	to make sure we have linear spacing

        print("-----Convolution with LSF-----")

        delta_wave = np.mean(self.wavelen[1:] - self.wavelen[:-1])
        print("step in wavelen:", delta_wave)

        # what if the input is not linear in wavelength?
        # should we interpolate on the output grid first?

        deltav = const.c * delta_wave/np.mean(self.wavelen)

        print("step in velocity:", deltav)

	# FWHM=3km/s, see wikipedia.org/wiki/Gaussian_function
        stddev = 3.*u.km/u.s / (2.*np.sqrt(2.*np.log(2.)))
        stddev /= deltav
        stddev = stddev.to(u.dimensionless_unscaled)
        print("deltav =", deltav, "=> stddev =", stddev, "pixel")

        gauss = ac.Gaussian1DKernel(stddev)

        for i_x in range(self.target_cube.shape[2]):
            print(self.target_cube.shape[2]-i_x, end=' ', flush=True)
            for i_y in range(self.target_cube.shape[1]):
                self.target_cube[:,i_y,i_x] = ac.convolve_fft(self.target_cube[:,i_y,i_x],
                                                              gauss, boundary='wrap')

        print()
        if plot:
            plt.plot(self.target_cube[:,111,100])
            plt.title("Pixel [111,100] after convolution with LSF")
            plt.show()


    #############################################################################
    def scale_to_detector(self):
        '''
        Scale to Detector pixels (spatially and spectrally)
        '''

        # First interpolate spectrally

        naxis3, naxis2, naxis1 = self.target_cube.shape

        wavelen = self.wavelen * u.um	# store units in self?
        in_velos = wavelen.to(u.m/u.s, equivalencies=u.doppler_optical(self.restcoo))

        #print("Original velos:", in_velos[0], "...", in_velos[-1])

        out_velos = self.det_velocities
        #print("New velos:", out_velos.shape, out_velos[0], "...", out_velos[-1])
        #print(out_velos)

        scaled_cube = np.empty((len(out_velos), naxis2, naxis1), self.target_cube[0,0,0].dtype)

        for i_x in range(naxis2):
            #print(i_x, end=' ',flush=True)
            for i_y in range(naxis1):
                intpol = interp1d(in_velos, self.target_cube[:, i_y, i_x],
                                  kind='linear', bounds_error=False, fill_value=0.)
                scaled_cube[:, i_y, i_x] = intpol(out_velos)
        print()
        self.target_cube = scaled_cube

        self.wcs.wcs.ctype[2] = 'VELO'
        self.wcs.wcs.crpix[2] = 1
        self.wcs.wcs.crval[2] = out_velos[0]
        self.wcs.wcs.cdelt[2] = out_velos[1]-out_velos[0]	# should be 1.5km/s
        self.wcs.wcs.cunit[2] = 'm/s'
        #
        # Now interpolate spatially
        #
        naxis3, naxis2, naxis1 = self.target_cube.shape
        print("ScaleToDetector: naxis =", naxis1, naxis2, naxis3)

        in_x = np.arange(naxis1)
        in_y = np.arange(naxis2)
        #print(len(in_x))

        # TODO better: scale from the center of the img

        print("image pixscale from WCS:", self.src_pixscale, "/pixel")

        scale = self.src_pixscale.value / self.det_pixscale
        print("Scale factor:", scale)

        # we need to scale the coord of the last pixel, not the pixel behind the end!
        out_x = np.arange(round((naxis1-1)*scale[0])+1) / scale[0]
        out_y = np.arange(round((naxis2-1)*scale[1])+1) / scale[1]
        print(len(out_x), len(out_y))

        scaled_cube = np.empty((naxis3, len(out_y), len(out_x)), self.target_cube[0,0,0].dtype)

        for i in range(naxis3):
            # bilinear interpol
            interp = RectBivariateSpline(in_x, in_y, self.target_cube[i,:,:], kx=1, ky=1)
            scaled_cube[i,:,:] = interp(out_x, out_y, grid=True)

        print("ScaleToDetector: new shape =", scaled_cube.shape)

        self.target_cube = scaled_cube

        self.wcs.wcs.cdelt[0] = -self.det_pixscale/3600./1000.	# convert mas/pix to deg/pix
        self.wcs.wcs.cdelt[1] =  self.det_pixscale/3600./1000.
        self.wcs.wcs.cunit[0] = 'deg'
        self.wcs.wcs.cunit[1] = 'deg'

	# overwrite old WCS, but keep rest of the header
        for card in self.wcs.to_header().cards:
            self.target_hdr[card.keyword] = (card.value, card.comment)


    #############################################################################

    def compute_snr(self, integration_time, write_src_w_bg=None, write_background=None):
        '''Compute SNR of the simulated observation (step 4 of the big plan)'''

        mirr_list = self.cmds.mirrors_telescope
        mirr_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                       mirr_list["Inner"]**2) * u.m**2
        print("Collecting mirror area: ", mirr_area)

        ph_cube = self.target_cube * u.Jy * mirr_area
	# per arcsec^2, but astropy cannot convert that
        ph_cube = ph_cube.to(u.photon / (u.s * u.um),
                             equivalencies=u.spectral_density(self.restcoo)) / (u.arcsec**2)

        print("Source peak:", np.max(ph_cube))
        print(ph_cube.shape)

        # wavelength of new target_cube frames in micron (not the same as self.wavelen!)
        wavelen = (self.det_velocities
                   * u.m/u.s).to(u.um, equivalencies=u.doppler_optical(self.restcoo))
        print("Wavelength grid:", wavelen.shape)

        backgrnd = self.emission * u.photon / (u.s*u.um*u.arcsec**2)
        print("Background max:", np.max(backgrnd))

        # dLambda = lambda * dv/c
        d_lambda = (4.8 * u.um * (1.5 *u.km/u.s) / const.c).to(u.um)	# in micron
        print("d_lambda", d_lambda)

        pix_area = (self.det_pixscale/1000. * u.arcsec)**2
        print("Pixel area", pix_area)

        ph_cube *= d_lambda*pix_area
        backgrnd *= d_lambda*pix_area

        print("peak pos:", np.unravel_index(np.argmax(ph_cube), ph_cube.shape))

        print("Source peak:", np.max(ph_cube))
        print("Background: ", np.max(backgrnd))

        #self.emission*d_lambda*, "ph/s/pix")
        plt.plot(wavelen, ph_cube[:,52,51])
        plt.plot(wavelen, backgrnd)
        plt.title("Pixel [52,51] in photons/sec")
        #plt.show()
        #
        # NOTE: SimMETIS.OpticalTrain includes the QE of the detector
        #       our units are actually electrons/s
        #
        bg_cube = backgrnd[:, np.newaxis, np.newaxis]
        bg_cube = np.tile(bg_cube, (1, ph_cube.shape[1], ph_cube.shape[2]))

        # bring the photons together (Isn't it romantic?)

        ph_cube += bg_cube

        plt.plot(wavelen, ph_cube[:,52,51])
        plt.title("Pixel [52,51] with background")
        plt.show()

        # Decide on the DIT to use, full well = 100e3, we fill to 80%

        peak = np.max(ph_cube)
        dit = 80e3 * u.photon / peak
        print("Peak", peak, ", using DIT", dit)

        ph_cube *= dit
        bg_cube *= dit

        print("Peak in one DIT", np.max(ph_cube))

        targ_noise = np.sqrt(ph_cube*u.photon + (70 * u.photon)**2)	# RON = 70e/pix/read
        back_noise = np.sqrt(bg_cube*u.photon + (70 * u.photon)**2)	# RON = 70e/pix/read

        ndit = np.round(integration_time / dit)
        print("Total integration time", integration_time, "=> NDIT =", ndit)

        ph_cube *= ndit
        bg_cube *= ndit
        targ_noise *= np.sqrt(ndit)
        back_noise *= np.sqrt(ndit)

        # multiply by normal distribution of stddev=1
        # => normal distributed errors with stddev=noise
        targ_noise *= np.random.standard_normal(size=targ_noise.shape)
        back_noise *= np.random.standard_normal(size=back_noise.shape)

        ph_cube += targ_noise
        bg_cube += back_noise

        header = self.target_hdr.copy()
        header['BUNIT'] = "e/pixel"

        if write_src_w_bg is not None:
            hdu = fits.PrimaryHDU(ph_cube.value, header=header)
            hdu.writeto(write_src_w_bg, overwrite=True)

        if write_background is not None:
            hdu = fits.PrimaryHDU(bg_cube.value, header=header)
            hdu.writeto(write_background, overwrite=True)

        ph_cube -= bg_cube
        hdu = fits.PrimaryHDU(ph_cube.value, header=header)

        #hdu.writeto("test_src-bg.fits", overwrite=True)

        return hdu


    #############################################################################

    def save_cube(self, outname):
        '''write the data cube to a fits-file'''

        hdu = fits.PrimaryHDU(self.target_cube, header=self.target_hdr)
        hdu.writeto(outname, overwrite=True)


    #############################################################################

    def simulate(self, conditions, psf_name, integration_time, plot=False):
        '''run a LMS simulation'''

        self.transmission_emission(conditions=conditions, plot=plot)
        self.convolve_psf(psf_name, plot=plot)
        self.convolve_lsf(plot=plot)
        self.scale_to_detector()
        self.compute_snr(integration_time)


#############################################################################
