#!/usr/bin/env python3
# coding=latin1
"""
spectro.py
Created:     Sat Oct 27 14:52:39 2018 by Koehler@Quorra
Last change: Sat Nov  3 14:21:41 2018

Python-script to simulate LMS of METIS

TODO: easy switch for best/median/poor conditions
"""

import sys
import numpy as np

from scipy.interpolate import interp1d, RectBivariateSpline

from astropy import units as u
from astropy.io import fits
from astropy import constants as const
from astropy import wcs
import astropy.convolution as ac

import matplotlib.pyplot as plt

import simmetis as sm

#############################################################################

def scale_image(img, scale):
    '''interpolate image to new shape'''

    naxis2, naxis1 = img.shape
    print("scale_image: naxis =", naxis1, naxis2)

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
    lam0: wavelength at zero-point of 3.dimension
    """

    def __init__(self, filename, lam0):
        '''Read datacube and WCS'''

        self.filename = filename
        self.lam0 = lam0

        print("Source file ", filename)

        self.src_fits = fits.open(filename)
        self.src_cube = self.src_fits[0].data
        self.src_header = self.src_fits[0].header

        naxis3, naxis2, naxis1 = self.src_cube.shape
        print(naxis1, "x", naxis2, "pixels,", naxis3, "velocities")

        # Parse the WCS keywords in primary HDU
        self.wcs = wcs.WCS(self.src_header)

        self.emission = None	# will be computed later


    def transmission_emission(self, skyfile='skycal_R308296_best_conditions.fits',
                              plot=False):
        '''Apply transission to src_cube and calculate emission spectrum'''

        # TODO: choose between best/median/poor

        print("-----Transmission and Emission-----")
        naxis3 = self.src_cube.shape[0]

        # make a Nlambda x 3 array
        crd = np.zeros((naxis3, 3))
        crd[:,2] = np.arange(naxis3)

        # Convert pixel coordinates to world coordinates
        # Second argument is "origin" -- in this case 0-based coordinates
        velos = self.wcs.wcs_pix2world(crd, 0)[:,2] * self.wcs.wcs.cunit[2]
        # should be u.m/u.s

        print("CTYPES:", self.wcs.wcs.ctype)

        # TODO: implement other CTYPES

        if self.wcs.wcs.ctype[2] != 'VELO':
            raise NotImplementedError('spectral axis must have type VELO')

        #print("Velocities:",velos)
        wavelen = self.lam0 * (1. + velos / const.c)	# in same units as lam0, hopefully micron
        #print("Wavelengths:",wavelen)

        #plt.plot(wavelen)
        #plt.show()
        if plot:
            print("Plotting pixel [111,100] from source cube")
            plt.plot(wavelen, self.src_cube[:,111,100])
            plt.show()

        #############################################################################
        # Read in trans-/emission from skycal

        skytransfits = fits.open(skyfile)
        print("Reading ", skyfile)

        skytrans = skytransfits[1].data

        skylam = skytrans['LAMBDA']
        skytran = skytrans['TRANS']
        skyemis = skytrans['EMIS']

        #plt.plot(skylam, skytran)
        #plt.show()

        idx = (np.where((skylam >= np.min(wavelen)) & (skylam <= np.max(wavelen))))[0]
        print("Index skytran to src-wave:", idx[0], "...", idx[-1])
        #
        # interpolate transmission/emission onto source-grid
        #
        transmission = np.interp(wavelen, skylam, skytran) # dimensionless, [0...1]
        emission     = np.interp(wavelen, skylam, skyemis) # photons/s/um/m^2/arcsec^2

        if plot:
            print("Plotting sky transmission in source wavelength range...")
            plt.plot(skylam[idx], skytran[idx], "+")
            plt.plot(wavelen, transmission)
            #plt.show()

        #############################################################################
        # get trans-/emission from SimMETIS
        #
        # TODO: set best/median/poor conditions in optical train
        #
        print("Firing optical train")

        cmds = sm.UserCommands('metis_image_generic.config')

        cmds["SIM_VERBOSE"] = 'hell no!'
        cmds["INST_FILTER_TC"] = "./TC_filter_open.dat"
	# open filter, but remember that the detector cuts off at 5.5um

        cmds["SIM_DETECTOR_PIX_SCALE"] = 1.
        # We set the pixel size to 1 arcsec^2, this will give us emission per arcsec^2

        # Create transmission curve.
        opttrain = sm.OpticalTrain(cmds)

        #print(cmds["INST_FILTER_TC"])

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
        #		 (i,e. _without_ the atmosphere)
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

        idx = (np.where((tc_lam >= np.min(wavelen)) & (tc_lam <= np.max(wavelen))))[0]
        print("Index SimMetis tc_lam to src-wave:", idx)

        #print("Plotting SimMetis-transmission...")
        #plt.plot(tc_lam,tc_val)
        #plt.show()

        if plot:
            print("Plotting SimMetis-transmission in source wavelength range...")
            plt.plot(wavelen, tc_trans(wavelen))
            plt.show()

        transmission *= tc_trans(wavelen)

        #############################################################################
        #
        # transmit the source cube!
        #
        self.src_cube = self.src_cube * transmission[:, np.newaxis, np.newaxis]

        if plot:
            print("Plotting pixel [111,100] from transmitted cube")
            plt.plot(wavelen, self.src_cube[:,111,100])
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

        ph_mirror = interp1d(lam, ph_mirr_um, kind='linear', bounds_error=False, fill_value=0.)

        print("Pix_res: ", opttrain.cmds.pix_res)

        mirr_list = cmds.mirrors_telescope
        mirr_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                       mirr_list["Inner"]**2)
        print("Mirror area: ", mirr_area)

        ph_atmo_um = emission * mirr_area * tc_trans(wavelen)
        if plot:
            print("Plotting atmospheric & mirror emission...")
            plt.plot(wavelen, ph_atmo_um, '+')
            plt.plot(wavelen, ph_mirror(wavelen))
            plt.plot(wavelen, ph_atmo_um+ph_mirror(wavelen))
            plt.show()

        self.emission = ph_atmo_um+ph_mirror(wavelen)


    #############################################################################

    def convolve_psf(self, psf_name, plot=False):
        '''Convolve src_sube with Point Spread Function'''

        print("-----Convolution with PSF-----")

        img_pixscale = wcs.utils.proj_plane_pixel_scales(self.wcs)[0:2]*3600.*1000.
        print("pixscale from WCS:", img_pixscale)
        if img_pixscale[0] != img_pixscale[1]:
            print("not square!")

        # TODO: Use WCS to get pixel scale?
        imgpixscale = self.src_header['PIXSCALE']*1000.
        print("Image pixel scale is", imgpixscale, "mas/pix")

        psf_fits = fits.open(psf_name)

        ext = psf_fits.index_of("PSF_3.80UM")
        print("PSF for 3.8um is extension", ext)
        psf_img = psf_fits[ext].data
        psf_hdr = psf_fits[ext].header

        psf_pixscale = psf_hdr['PIXSCALE']
        print("PSF pixel scale is", psf_pixscale, "mas/pix @ 3.8 um,", psf_img.shape, "pixels")

        psf_pixscale *= self.lam0 / 3.8
        print("PSF pixel scale is", psf_pixscale, "mas/pix @", self.lam0, "um,")

        scale = psf_pixscale/imgpixscale
        print("Scale factor", scale)

        psf_scaled = scale_image(psf_img, scale)

        norm = np.sum(psf_scaled)
        print("Normalizing PSF by factor", norm)
        psf_scaled /= norm

        psf_hdr['EXTNAME'] = None
        hdu = fits.PrimaryHDU(psf_scaled)
        hdu.header['WAVELENG'] = self.lam0
        hdu.header['PIXSCALE'] = psf_pixscale
        hdu.writeto("PSF_scaled.fits", overwrite=True)

        #############################################################################
        print("Convolving with PSF...")
        print("Frame ", end=' ', flush=True)

        for l in range(self.src_cube.shape[0]):
            print(l, end=' ', flush=True)
            self.src_cube[l,:,:] = ac.convolve_fft(self.src_cube[l,:,:], psf_scaled)

        print()
        if plot:
            plt.plot(self.src_cube[:,111,100])


    #############################################################################
    def convolve_lsf(self, plot=False):
        '''Convolve with Line Spread Function'''

        print("-----Convolution with LSF-----")

        cdelt3 = self.wcs.wcs.cdelt[2]
        cunit3 = self.wcs.wcs.cunit[2]
        deltav = cdelt3 * cunit3

	# FWHM=3km/s, see wikipedia.org/wiki/Gaussian_function
        stddev = 3.*u.km/u.s / (2.*np.sqrt(2.*np.log(2.)))
        stddev /= deltav
        stddev = stddev.to(u.dimensionless_unscaled)
        print("deltav =", deltav, "=> stddev =", stddev, "pixel")
        #print("deltav =", deltav, "km/s => stddev =", stddev, "pixel")

        gauss = ac.Gaussian1DKernel(stddev)

        for i_x in range(self.src_cube.shape[2]):
            print(i_x, end=' ', flush=True)
            for i_y in range(self.src_cube.shape[1]):
                self.src_cube[:,i_y,i_x] = ac.convolve_fft(self.src_cube[:,i_y,i_x],
                                                           gauss, boundary='wrap')

        print()
        if plot:
            plt.plot(self.src_cube[:,111,100])
            plt.show()

    #############################################################################
    def scale_to_detector(self, detector_pixscale=8.2):
        '''
        Scale to Detector pixels (spatially and spectrally)
        Parameter detector_pixscale is in mas/pixel (default 8.2)
        '''

        # first interpolate spectrally,
        # (the spatial interpolation will increase the amount of data)

        naxis3, naxis2, naxis1 = self.src_cube.shape
        crd = np.zeros((naxis3, 3))	# make a Nlambda x 3 array
        crd[:,2] = np.arange(naxis3)

        # Convert pixel coordinates to world coordinates
        # Second argument is "origin" -- in this case 0-based coordinates
        in_velos = self.wcs.wcs_pix2world(crd, 0)[:,2] / 1000.	# make it km/s

        #print("Original velos:")
        #print(in_velos)

        step = 1.5	# u.km/u.s

        new3 = int((in_velos[-1] - in_velos[0]) / step)+1
        print("new naxis3:", new3)

        meanv = (in_velos[0] + in_velos[-1]) / 2.
        print("mean v:", meanv)

        out_velos = np.arange(meanv-step*(new3-1)/2., meanv+step*((new3-1)/2.+1), step)
        #print("New velos:",out_velos.shape)
        #print(out_velos)

        scaled_cube = np.empty((len(out_velos), naxis2, naxis1), self.src_cube[0,0,0].dtype)

        for i_x in range(naxis2):
            #print(i_x,end=' ',flush=True)
            for i_y in range(naxis1):
                intpol = interp1d(in_velos, self.src_cube[:, i_y, i_x],
                                  kind='linear', bounds_error=False, fill_value=0.)
                scaled_cube[:, i_y, i_x] = intpol(out_velos)
        print()
        self.src_cube = scaled_cube

        self.wcs.wcs.cdelt[2] = step*1000.
        self.wcs.wcs.cunit[2] = 'm/s'
        #
        # Now interpolate spatially
        #
        naxis3, naxis2, naxis1 = self.src_cube.shape
        print("ScaleToDetector: naxis =", naxis1, naxis2, naxis3)

        in_x = np.arange(naxis1)
        in_y = np.arange(naxis2)
        #print(len(in_x))

        # TODO better: scale from the center of the img
        # TODO: get pixel scale from WCS

        imgpixscale = self.src_header['PIXSCALE']*1000.
        print("Image pixel scale is", imgpixscale, "mas/pix")

        scale = detector_pixscale / imgpixscale	# Detector has 8.2 mas/pix (?)

        # we need to scale the coord of the last pixel, not the pixel behind the end!
        out_x = np.arange(round((naxis1-1)*scale)+1) / scale
        out_y = np.arange(round((naxis2-1)*scale)+1) / scale
        #print(len(out_x))

        scaled_cube = np.empty((naxis3, len(out_y), len(out_x)), self.src_cube[0,0,0].dtype)

        for l in range(naxis3):
            # bilinear interpol
            interp = RectBivariateSpline(in_x, in_y, self.src_cube[l,:,:], kx=1, ky=1)
            scaled_cube[l,:,:] = interp(out_x, out_y, grid=True)

        print("ScaleToDetector: new shape =", scaled_cube.shape)

        self.src_cube = scaled_cube

        self.wcs.wcs.cdelt[0] = -detector_pixscale/3600./1000.	# convert mas/pix to deg/pix
        self.wcs.wcs.cdelt[1] =  detector_pixscale/3600./1000.
        self.wcs.wcs.cunit[0] = 'deg'
        self.wcs.wcs.cunit[1] = 'deg'

        self.src_header = self.wcs.to_header()


    #############################################################################

    def save_cube(self, outname):
        '''write the data cube to a fits-file'''

        hdu = fits.PrimaryHDU(self.src_cube, header=self.src_header)
        hdu.writeto(outname, overwrite=True)


#############################################################################

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("USAGE: ", sys.argv[0], " filename lambda0 [plot]\n")
        exit(5)

    lms = LMS(sys.argv[1], float(sys.argv[2]))
    lms.transmission_emission(plot=(len(sys.argv) > 3))

    lms.convolve_psf("metis_psf_mag=08.00_seeing=1.00.fits", plot=(len(sys.argv) > 3))
    lms.save_cube("test_conv_PSF.fits")

    lms.convolve_lsf(plot=(len(sys.argv) > 3))
    lms.save_cube("test_conv_LSF.fits")

    #lms = LMS("test_conv_LSF.fits", float(sys.argv[2]))

    lms.scale_to_detector()
    lms.save_cube("test_detector.fits")
