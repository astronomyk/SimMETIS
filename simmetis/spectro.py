#!/usr/bin/env python3
# coding=latin1
"""
spectro.py
Created:     Sat Oct 27 14:52:39 2018 by Koehler@Quorra
Last change: Mon Nov  5 09:37:50 2018

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
        print("Source file ", filename)

        self.src_fits = fits.open(filename)
        self.src_cube = self.src_fits[0].data
        self.src_header = self.src_fits[0].header

        naxis3, naxis2, naxis1 = self.src_cube.shape
        print(naxis1, "x", naxis2, "pixels,", naxis3, "velocities")

        # Parse the WCS keywords in primary HDU
        self.wcs = wcs.WCS(self.src_header)

        pixscale1, pixscale2 = wcs.utils.proj_plane_pixel_scales(self.wcs)[0:2]
        pixscale1 = (pixscale1 * self.wcs.wcs.cunit[0]).to(u.mas)
        pixscale2 = (pixscale2 * self.wcs.wcs.cunit[1]).to(u.mas)
        #print("image pixscale from WCS:", pixscale1, pixscale2, "/pixel")

        self.src_pixscale = np.asarray((pixscale1.value, pixscale2.value))*u.mas
        print("image pixscale:", self.src_pixscale, "per pixel")

        # make a Nlambda x 3 array
        crd = np.zeros((naxis3, 3))
        crd[:,2] = np.arange(naxis3)

        # Convert pixel coordinates to world coordinates
        # Second argument is "origin" -- in this case 0-based coordinates
        world_coo = self.wcs.wcs_pix2world(crd, 0)[:,2] * self.wcs.wcs.cunit[2]

        print("CTYPES:", self.wcs.wcs.ctype)

        # TODO: implement other CTYPES

        if self.wcs.wcs.ctype[2] == 'VELO':
            #print("Velocities:",world_coo)
            # should be u.m/u.s
            wavelen = lam0 * (1. + world_coo / const.c)	 # in same units as lam0, hopefully micron
            #print("Wavelengths:",wavelen)
        elif self.wcs.wcs.ctype[2] == 'FREQ':
            wavelen = world_coo.to(u.um, equivalencies=u.spectral())
        else:
            raise NotImplementedError('spectral axis must have type VELO or FREQ')

        print("Wavelengths:", wavelen[0:3], "...", wavelen[-1])
        self.wavelen = wavelen.value

        self.emission = None	# will be computed later


    def transmission_emission(self, skyfile='skycal_R308296_best_conditions.fits',
                              plot=False):
        '''Apply transission to src_cube and calculate emission spectrum'''

        # TODO: choose between best/median/poor

        print("-----Transmission and Emission-----")
        if plot:
            print("Plotting pixel [111,100] from source cube")
            plt.plot(self.wavelen, self.src_cube[:,111,100])
            plt.title("Pixel [111,100] in source cube")
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

        idx = (np.where((skylam >= np.min(self.wavelen)) & (skylam <= np.max(self.wavelen))))[0]
        print("Index skytran to src-wave:", idx[0], "...", idx[-1])
        #
        # interpolate transmission/emission onto source-grid
        #
        transmission = np.interp(self.wavelen, skylam, skytran) # dimensionless, [0...1]
        emission     = np.interp(self.wavelen, skylam, skyemis) # photons/s/um/m^2/arcsec^2

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

        idx = (np.where((tc_lam >= np.min(self.wavelen)) & (tc_lam <= np.max(self.wavelen))))[0]
        print("Index SimMetis tc_lam to src-wave:", idx)

        #print("Plotting SimMetis-transmission...")
        #plt.plot(tc_lam,tc_val)
        #plt.show()

        if plot:
            print("Plotting SimMetis-transmission in source wavelength range...")
            plt.plot(self.wavelen, tc_trans(self.wavelen))
            plt.show()

        transmission *= tc_trans(self.wavelen)

        #############################################################################
        #
        # transmit the source cube!
        #
        self.src_cube = self.src_cube * transmission[:, np.newaxis, np.newaxis]

        if plot:
            print("Plotting pixel [111,100] from transmitted cube")
            plt.plot(self.wavelen, self.src_cube[:,111,100])
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

        ph_mirror = interp1d(lam, ph_mirr_um, kind='linear', bounds_error=False, fill_value=0.)

        print("Pix_res: ", opttrain.cmds.pix_res)

        mirr_list = cmds.mirrors_telescope
        mirr_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                       mirr_list["Inner"]**2)
        print("Mirror area: ", mirr_area)

        ph_atmo_um = emission * mirr_area * tc_trans(self.wavelen)
        if plot:
            print("Plotting atmospheric & mirror emission...")
            plt.plot(self.wavelen, ph_atmo_um, '+')
            plt.plot(self.wavelen, ph_mirror(self.wavelen))
            plt.plot(self.wavelen, ph_atmo_um+ph_mirror(self.wavelen))
            plt.title("Emission")
            plt.show()

        self.emission = ph_atmo_um+ph_mirror(self.wavelen)


    #############################################################################

    def convolve_psf(self, psf_name, plot=False):
        '''Convolve src_sube with Point Spread Function'''

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
        print("PSF pixel scale is", psf_pixscale, "mas/pix @ 3.8 um,", psf_img.shape, "pixels")

        psf_pixscale *= np.mean(self.wavelen) / 3.8
        print("PSF pixel scale is", psf_pixscale, "mas/pix @", np.mean(self.wavelen), "um,")

        scale = psf_pixscale / self.src_pixscale[0].value
        print("Scale factor", scale)

        psf_scaled = scale_image(psf_img, scale)

        norm = np.sum(psf_scaled)
        print("Normalizing PSF by factor", norm)
        psf_scaled /= norm

        psf_hdr['EXTNAME'] = None
        hdu = fits.PrimaryHDU(psf_scaled)
        hdu.header['WAVELENG'] = np.mean(self.wavelen)
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
            plt.title("Pixel [111,100] after convolution with LSF")
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

        in_velos = const.c * (self.wavelen/np.mean(self.wavelen) - 1.)

        print("Original velos:", in_velos[0], "...", in_velos[-1])

        step = 1.5 * u.km/u.s

        new3 = int((in_velos[-1] - in_velos[0]) / step)+1
        print("new naxis3:", new3)

        meanv = (in_velos[0] + in_velos[-1]) / 2.
        print("mean v:", meanv)
        #meanv = meanv.value

        start = (meanv-step*(new3-1)/2.).to(u.m/u.s).value
        end   = (meanv+step*((new3-1)/2.+1)).to(u.m/u.s).value
        step  = step.to(u.m/u.s).value

        out_velos = np.arange(start, end, step)
        print("New velos:", out_velos.shape)
        print(out_velos)

        scaled_cube = np.empty((len(out_velos), naxis2, naxis1), self.src_cube[0,0,0].dtype)

        for i_x in range(naxis2):
            #print(i_x,end=' ',flush=True)
            for i_y in range(naxis1):
                intpol = interp1d(in_velos, self.src_cube[:, i_y, i_x],
                                  kind='linear', bounds_error=False, fill_value=0.)
                scaled_cube[:, i_y, i_x] = intpol(out_velos)
        print()
        self.src_cube = scaled_cube

        self.wcs.wcs.ctype[2] = 'VELO'
        self.wcs.wcs.crpix[2] = 1
        self.wcs.wcs.crval[2] = start
        self.wcs.wcs.cdelt[2] = step
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

        print("image pixscale from WCS:", self.src_pixscale, "/pixel")

        scale = detector_pixscale / self.src_pixscale.value
        print("Scale factor:", scale)

        # we need to scale the coord of the last pixel, not the pixel behind the end!
        out_x = np.arange(round((naxis1-1)*scale[0])+1) / scale[0]
        out_y = np.arange(round((naxis2-1)*scale[1])+1) / scale[1]
        print(len(out_x), len(out_y))

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

    #lms.convolve_psf("metis_psf_mag=08.00_seeing=1.00.fits", plot=(len(sys.argv) > 3))
    #lms.save_cube("test_conv_PSF.fits")

    #lms.convolve_lsf(plot=(len(sys.argv) > 3))
    #lms.save_cube("test_conv_LSF.fits")

    #lms = LMS("test_conv_LSF.fits", float(sys.argv[2]))

    lms.scale_to_detector()
    lms.save_cube("test_detector.fits")
