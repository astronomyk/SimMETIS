"""
spectro.py
Created:     Sat Oct 27 14:52:39 2018 by Koehler@Quorra
Last change: Wed Dec  5 17:10:21 2018

Python-script to simulate LMS of METIS
"""

from datetime import datetime

import numpy as np

from scipy.interpolate import interp1d, RectBivariateSpline

from astropy import units as u
from astropy.io import fits
from astropy import constants as const
from astropy import wcs
import astropy.convolution as ac

import matplotlib
import matplotlib.pyplot as plt

import simmetis as sm
import simmetis.spectral as sc

import warnings

#############################################################################

def _scale_image(img, scale, verbose=False):
    '''interpolate image to new shape'''

    naxis2, naxis1 = img.shape
    if verbose:
        print("scale_image: old shape =", img.shape)

    in_x = np.arange(naxis1)
    in_y = np.arange(naxis2)
    #print(len(in_x))

    # this scales from pixel 0,0
    #out_x = np.arange(round((naxis1-1)*scale)+1) / scale
    #out_y = np.arange(round((naxis2-1)*scale)+1) / scale
    #print(len(out_x))

    # scale from the center of the image
    half1 = naxis1//2
    half2 = naxis2//2

    # scale the coord of the last pixel, not the pixel behind the end!
    out_x = np.arange(half1 - round(half1*scale)/scale,
                      half1 + round((half1-1)*scale)/scale + 1, 1./scale)
    out_y = np.arange(half2 - round(half2*scale)/scale,
                      half2 + round((half2-1)*scale)/scale + 1, 1./scale)

    interp = RectBivariateSpline(in_x, in_y, img, kx=1, ky=1)	# bilinear interpol
    scaled_img = interp(out_x, out_y, grid=True)

    if verbose:
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

    def __init__(self, filename, config, lambda0=3.8, verbose=False):
        '''Read datacube and WCS'''

        self.wide_figsize = matplotlib.rcParams['figure.figsize'].copy()
        self.wide_figsize[0] = self.wide_figsize[1]*16./9.
        self.verbose = verbose

	# separate our output from the bullshit printed by import
        print('============================================')

        self.filename = filename
        print("Source file ", filename)

        self.src_fits = fits.open(filename)
        self.src_cube = self.src_fits[0].data
        self.src_header = self.src_fits[0].header

        naxis3, naxis2, naxis1 = self.src_cube.shape
        print(naxis1, "x", naxis2, "pixels,", naxis3, "spectral channels")

        self.plotpix = np.asarray((naxis2//2, naxis1//2))

        # Parse the WCS keywords in primary HDU
        self.wcs = wcs.WCS(self.src_header)

        pixscale1, pixscale2 = wcs.utils.proj_plane_pixel_scales(self.wcs)[0:2]
        pixscale1 = (pixscale1 * self.wcs.wcs.cunit[0]).to(u.mas)
        pixscale2 = (pixscale2 * self.wcs.wcs.cunit[1]).to(u.mas)
        #print("image pixscale from WCS:", pixscale1, pixscale2, "/pixel")

        self.src_pixscale = np.asarray((pixscale1.value, pixscale2.value))*u.mas
        if self.verbose:
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
                if self.verbose:
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

        if self.verbose:
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

        self.wavelen = wavelen.value	# should we store it with units?
        self.restcoo = restcoo		# RESTFRQ or RESTWAV with units

        if self.verbose:
            print("Wavelengths:", wavelen[0:3], "...", wavelen[-1])
            print("restfrq:", self.wcs.wcs.restfrq)
            print("restwav:", self.wcs.wcs.restwav)
            #print("Rest wavelen:", restcoo.to(u.um, equivalencies=u.spectral()))

        in_velos = wavelen.to(u.m/u.s, equivalencies=u.doppler_optical(restcoo))
        step = 1.5 * u.km/u.s
        new3 = int((in_velos[-1] - in_velos[0]) / step)+1
        meanv = (in_velos[0] + in_velos[-1]) / 2.

        self.det_velocities = np.arange((meanv-step*(new3-1)/2.).to(u.m/u.s).value,
                                        (meanv+step*((new3-1)/2.+1)).to(u.m/u.s).value,
                                        step.to(u.m/u.s).value)

        if self.verbose:
            print("Source velocities (WCS):", in_velos[0], "...", in_velos[-1])
            #print(in_velos)
            print("new naxis3:", new3)
            #print("middle v:", meanv)
            print("Detector velocities:", self.det_velocities.shape,
                  self.det_velocities[0], "...", self.det_velocities[-1])

        # initialize the target cube, in case someone does not call transmission_emission()

        self.target_cube = self.src_cube
        self.target_hdr = self.src_header.copy()
        self.target_hdr['BUNIT'] = 'Jy/arcsec2'
        self.background = None

        self.transmission = 1.
        self.emission = None	# will be computed later

        # initialize SimCADO to set searchpaths
        #
        print("Reading config ", config)
        self.cmds = sm.UserCommands(config)

        self.det_pixscale = self.cmds["SIM_DETECTOR_PIX_SCALE"] * 1000.	 # in mas/pixel

        if self.verbose:
            print("Detector pixel scale ", self.det_pixscale, " mas/pixel")
            print("Filter = ", self.cmds["INST_FILTER_TC"])	# should be open filter


    #############################################################################

    def add_cmds_to_header(self, header):
        '''
        Add user commands to Fits header
        '''
        header['DATE'] = datetime.now().isoformat()

        for key in self.cmds.cmds:
            val = self.cmds.cmds[key]

            if isinstance(val, (sc.TransmissionCurve, sc.EmissionCurve,
                                sc.UnityCurve, sc.BlackbodyCurve)):
                val = val.params["filename"]

            if isinstance(val, str) and len(val) > 35:
                val = "... " + val[-35:]

            header["HIERARCH "+key] = val



    #############################################################################

    def transmission_emission(self, conditions='median', plot=False):
        '''
        Apply transission to src_cube and calculate emission spectrum
        '''
        print()
        print("--------------------Transmission and Emission--------------------")
        if plot:
            plt.plot(self.wavelen, self.src_cube[:, self.plotpix[0], self.plotpix[1]])
            plt.title("Pixel ["+str(self.plotpix[0])+","+str(self.plotpix[1])+"] in source cube")
            plt.xlabel("Wavelength [micron]")
            plt.ylabel("Flux [Jy/arcsec2]")
            plt.show()

        # calculate wavelens of detector, needed for emission
        #
        det_wavelen = (self.det_velocities
                       * u.m/u.s).to(u.um, equivalencies=u.doppler_optical(self.restcoo))

        #############################################################################
        # Read in trans-/emission from skycal
        #
        if conditions == 'best':
            skyfile = 'skycal_R308296_best_conditions.fits'
            self.cmds["SCOPE_TEMPERATURE"] = 258.-273.
            self.cmds["SCOPE_MIRROR_LIST"] = "EC_mirrors_EELT_SCAO_best.tbl"
        elif conditions == 'median':
            skyfile = 'skycal_R308296_median_conditions.fits'
            self.cmds["SCOPE_TEMPERATURE"] = 282.-273.
            self.cmds["SCOPE_MIRROR_LIST"] = "EC_mirrors_EELT_SCAO_median.tbl"
        elif conditions == 'poor':
            skyfile = 'skycal_R308296_poor_conditions.fits'
            self.cmds["SCOPE_TEMPERATURE"] = 294.-273.
            self.cmds["SCOPE_MIRROR_LIST"] = "EC_mirrors_EELT_SCAO_poor.tbl"
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
        if self.verbose:
            print("Index skytran to src-wave:", idx[0], "...", idx[-1])
        #
        # interpolate transmission/emission onto source-grid
        #
        self.transmission = np.interp(self.wavelen, skylam, skytran) # dimensionless, [0...1]

        sky_emission = np.interp(det_wavelen, skylam, skyemis) * (self.det_pixscale/1000.)**2
        # emission in data file is photons/s/um/m^2/arcsec2, convert to photons/s/um/m^2

        if plot:
            plt.figure(num=1, figsize=self.wide_figsize)
            plt.subplots_adjust(left=0.1, right=0.75)
            plt.plot(skylam[idx], skytran[idx], "+", label='Sky transmission')
            plt.plot(self.wavelen, self.transmission, label='Sky tr. interpolated')
            plt.xlabel("Wavelength [micron]")
            plt.ylabel("Transmission")
            #plt.show()

        #############################################################################
        # get trans-/emission from SimMETIS
        #
        if self.verbose:
            print("Steaming up optical train")
            print("Telescope temperature:", self.cmds["SCOPE_TEMPERATURE"])

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
        tc_trans = interp1d(opttrain.tc_atmo.lam, opttrain.tc_atmo.val,
                            kind='linear', bounds_error=False, fill_value=0.)

        if plot:
            plt.plot(self.wavelen, tc_trans(self.wavelen), label='Tel. transmission')
            plt.title("Sky & Telescope Transmission")
            plt.xlabel("Wavelength [micron]")
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
            plt.show()

        # combine transmission of atmosphere, telescope, and LMS (the Roy factor)
        self.transmission *= tc_trans(self.wavelen) * 0.712599

        #############################################################################
        #
        # transmit the source cube!
        #
        self.target_cube = self.src_cube * self.transmission[:, np.newaxis, np.newaxis]
        self.add_cmds_to_header(self.target_hdr)

        # do not add skycal-file to self.cmds!
        # the optical train will be confused if we try to re-run it.
        self.target_hdr['SKYCAL_FILE'] = skyfile

        if plot:
            plt.plot(self.wavelen, self.target_cube[:, self.plotpix[0], self.plotpix[1]])
            plt.title("Pixel ["+str(self.plotpix[0])+","+str(self.plotpix[1])+"] in transmitted cube")
            plt.xlabel("Wavelength [micron]")
            plt.ylabel("Flux [Jy/arcsec2]")
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

        mirr_list = self.cmds.mirrors_telescope
        mirr_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                       mirr_list["Inner"]**2)
        if self.verbose:
            print("Pix_res: ", opttrain.cmds.pix_res)
            print("Mirror area: ", mirr_area, "[m^2]")

        ph_atmo_um = sky_emission * mirr_area * tc_trans(det_wavelen)
        ph_mirrors = ph_mirror(det_wavelen) * 1.25
        # Leo's fudge-factor to account for spiders and entrance window

        if plot:
            plt.figure(num=1, figsize=self.wide_figsize)
            plt.subplots_adjust(right=0.8)
            plt.plot(det_wavelen, ph_atmo_um, '+', label='Atmosphere')
            plt.plot(det_wavelen, ph_mirrors, label='Telescope')
            plt.plot(det_wavelen, ph_atmo_um+ph_mirrors, label='total')
            plt.title("Sky & Mirror Emission")
            plt.xlabel("Wavelength [micron]")
            plt.ylabel("Background Flux [photons/s/um/pixel]")
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
            plt.show()

        self.background = (ph_atmo_um+ph_mirrors) * 0.712599 	# photons/s/um/pixel
        # apply Roy factor for transmission of LMS

        if self.verbose:
            print("Final telescope temperature:", opttrain.cmds["SCOPE_TEMPERATURE"])



    #############################################################################

    def convolve_psf(self, psf_name, write_psf=None, plot=False):
        '''Convolve target_cube with Point Spread Function'''

        print()
        print("--------------------Convolution with PSF--------------------")

        print("image pixscale from WCS:", self.src_pixscale, "/ pixel")
        if self.src_pixscale[0] != self.src_pixscale[1]:
            print("pixels are not square!  Bad things will happen!")

        psf_name = sm.utils.find_file(psf_name)
        print("Reading PSF", psf_name)
        psf_fits = fits.open(psf_name)
        self.cmds['SCOPE_PSF_FILE'] = psf_name

        ext = psf_fits.index_of("PSF_3.80UM")
        if self.verbose:
            print("PSF for 3.8um is extension", ext)
        psf_img = psf_fits[ext].data
        psf_hdr = psf_fits[ext].header

        psf_pixscale = psf_hdr['PIXSCALE']
        if self.verbose:
            print("PSF pixel scale is",
                  psf_pixscale, "mas/pix @ 3.8 um,", psf_img.shape, "pixels")

        psf_pixscale *= np.mean(self.wavelen) / 3.8

        scale = psf_pixscale / self.src_pixscale[0].value
        if self.verbose:
            print("PSF pixel scale is", psf_pixscale, "mas/pix @", np.mean(self.wavelen), "um,")
            print("Scale factor", scale)

        psf_scaled = _scale_image(psf_img, scale)

        norm = np.sum(psf_scaled)
        if self.verbose:
            print("Normalizing PSF by factor", norm)
        psf_scaled /= norm

        # crop to size needed for input data
        # parameters found empirically to center peak of PSF
        # (would be better if _scale_image would scale from the center)

        psf_center = np.asarray(psf_scaled.shape) // 2
        img_shape = self.target_cube.shape[1:3]
        if self.verbose:
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
            #print(self.target_cube.shape[0]-i, end=' ', flush=True)
            print(" ", i*100//(self.target_cube.shape[0]-1), end='% \r', flush=True)
            self.target_cube[i,:,:] = ac.convolve_fft(self.target_cube[i,:,:], psf_scaled)
            # Note: convolve_fft sets values outside the image bounds to 0.

        print("       ", end='\r')
        if plot:
            plt.plot(self.wavelen, self.target_cube[:, self.plotpix[0], self.plotpix[1]])
            plt.title("Pixel ["+str(self.plotpix[0])+","+str(self.plotpix[1])+"] after convolution with PSF")
            plt.xlabel("Wavelength [micron]")
            plt.ylabel("Flux [Jy/arcsec2]")
            plt.show()


    #############################################################################
    def convolve_lsf(self, plot=False):
        '''Convolve with Line Spread Function'''

        # TODO: interpolate to velocity-grid before convolving
        #	to make sure we have linear spacing

        print()
        print("--------------------Convolution with LSF--------------------")

        # what if the input is not linear in wavelength?
        # should we interpolate on the output grid first?

        delta_wave = np.mean(self.wavelen[1:] - self.wavelen[:-1])
        deltav = const.c * delta_wave/np.mean(self.wavelen)

	# FWHM=3km/s, see wikipedia.org/wiki/Gaussian_function
        stddev = 3.*u.km/u.s / (2.*np.sqrt(2.*np.log(2.)))
        stddev /= deltav
        stddev = stddev.to(u.dimensionless_unscaled)

        if self.verbose:
            print("step in wavelen:", delta_wave)
            print("step in velocity:", deltav)
            print("      => stddev =", stddev, "pixel")

        gauss = ac.Gaussian1DKernel(stddev)

        for i_x in range(self.target_cube.shape[2]):
            #print(self.target_cube.shape[2]-i_x, end=' ', flush=True)
            if i_x % 2 == 0:
                print("\r ", i_x*100//(self.target_cube.shape[2]-1), end='% \r', flush=True)
            for i_y in range(self.target_cube.shape[1]):
                self.target_cube[:,i_y,i_x] = ac.convolve_fft(self.target_cube[:,i_y,i_x],
                                                              gauss, boundary='wrap')

        print("       ", end='\r')
        if plot:
            plt.plot(self.wavelen, self.target_cube[:, self.plotpix[0], self.plotpix[1]])
            plt.title("Pixel ["+str(self.plotpix[0])+","+str(self.plotpix[1])+"] after convolution with LSF")
            plt.xlabel("Wavelength [micron]")
            plt.ylabel("Flux [Jy/arcsec2]")
            plt.show()


    #############################################################################
    def scale_to_detector(self):
        '''
        Scale to Detector pixels (spatially and spectrally)
        '''

        print()
        print("--------------------Scale to detector pixels--------------------")

        # First interpolate spectrally

        naxis3, naxis2, naxis1 = self.target_cube.shape

        in_velos = (self.wavelen * u.um).to(u.m/u.s,
                                            equivalencies=u.doppler_optical(self.restcoo))

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
        #print()
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
        if self.verbose:
            print("ScaleToDetector: naxis =", naxis1, naxis2, naxis3)

        in_x = np.arange(naxis1)
        in_y = np.arange(naxis2)
        #print(len(in_x))

        # TODO better: scale from the center of the img

        scale = self.src_pixscale.value / self.det_pixscale

        if self.verbose:
            print("image pixscale from WCS:", self.src_pixscale, "/pixel")
            print("Scale factor:", scale)

        self.plotpix = np.rint(self.plotpix*scale).astype(int)

        ## we need to scale the coord of the last pixel, not the pixel behind the end!
        #out_x = np.arange(round((naxis1-1)*scale[0])+1) / scale[0]
        #out_y = np.arange(round((naxis2-1)*scale[1])+1) / scale[1]

        # scale from the center of the image
        half1 = naxis1//2
        half2 = naxis2//2

        # scale the coord of the last pixel, not the pixel behind the end!
        out_x = np.arange(half1 - round(half1*scale[1])/scale[1],
                          half1 + round((half1-1)*scale[1])/scale[1] + 1, 1./scale[1])
        out_y = np.arange(half2 - round(half2*scale[0])/scale[0],
                          half2 + round((half2-1)*scale[0])/scale[0] + 1, 1./scale[0])

        scaled_cube = np.empty((naxis3, len(out_y), len(out_x)), self.target_cube[0,0,0].dtype)

        for i in range(naxis3):
            # bilinear interpol
            interp = RectBivariateSpline(in_x, in_y, self.target_cube[i,:,:], kx=1, ky=1)
            scaled_cube[i,:,:] = interp(out_x, out_y, grid=True)

        if self.verbose:
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

    def compute_snr(self, exptime=0., ndit=0,
                    write_src_w_bg=None, write_background=None, plot=False):
        '''
        Compute SNR of the simulated observation (step 4 of the big plan)
        '''
        # TODO: give exptime as number (in sec) oder quantity (with user-given unit)

        if exptime > 0.:
            self.cmds["OBS_EXPTIME"] = exptime
        else:
            exptime = self.cmds["OBS_EXPTIME"]

        if ndit > 0:
            ndit = round(ndit)
            self.cmds["OBS_NDIT"] = ndit
        else:
            ndit = round(self.cmds["OBS_NDIT"])

        mirr_list = self.cmds.mirrors_telescope
        mirr_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                       mirr_list["Inner"]**2) * u.m**2
        if self.verbose:
            print("Collecting mirror area: ", mirr_area)

        ph_cube = self.target_cube * u.Jy * mirr_area
	# per arcsec^2, but astropy cannot convert that
        ph_cube = ph_cube.to(u.photon / (u.s * u.um),
                             equivalencies=u.spectral_density(self.restcoo)) / (u.arcsec**2)
        # technically, we convert to photo-electrons/s/um/arcsec2,
        # because the QE of the detector has been applied in the simmetis.OpticalTrain

        backgrnd = self.background * u.electron / (u.s*u.um)

        if self.verbose:
            print("Source peak:", np.max(ph_cube))
            print(ph_cube.shape)
            print("Background max:", np.max(backgrnd))

        # dLambda = lambda * dv/c
        d_lambda = (np.mean(self.wavelen)*u.um * (1.5 *u.km/u.s) / const.c).to(u.um)	# in micron
        pix_area = (self.det_pixscale/1000. * u.arcsec)**2

        if self.verbose:
            print("d_lambda", d_lambda)
            print("Pixel area", pix_area)

        ph_cube *= d_lambda*pix_area * u.electron/u.photon	# was electrons all the time
        backgrnd *= d_lambda	# per pixel

        if self.verbose:
            print("peak pos:", np.unravel_index(np.argmax(ph_cube), ph_cube.shape))

        print("Source peak:", np.max(ph_cube))
        print("Background: ", np.max(backgrnd))

        if plot:
            plt.figure(num=1, figsize=self.wide_figsize)
            plt.subplots_adjust(left=0.1, right=0.75)
            spectrum = ph_cube[:, self.plotpix[0], self.plotpix[1]]
            plt.plot(self.det_velocities * u.m/u.s, spectrum, label='Source')
            bf = 1
            while np.max(backgrnd)*bf*10 < np.max(spectrum):
                bf *= 10
            plt.plot(self.det_velocities * u.m/u.s, backgrnd*bf, label='Background*'+str(bf))
            plt.title("Source and background, pixel ["
                      +str(self.plotpix[0])+","+str(self.plotpix[1])+"]")
            plt.xlabel("Velocity [m/s]")
            plt.ylabel("Flux [e-/sec]")
            #plt.show()
        #
        # NOTE: SimMETIS.OpticalTrain includes the QE of the detector
        #       our units are actually electrons/s
        #
        bg_cube = backgrnd[:, np.newaxis, np.newaxis]
        bg_cube = np.tile(bg_cube, (1, ph_cube.shape[1], ph_cube.shape[2]))

        # bring the photons together (Isn't it romantic?)

        ph_cube += bg_cube

        if plot:
            plt.plot(self.det_velocities * u.m/u.s,
                     ph_cube[:, self.plotpix[0], self.plotpix[1]], label='Source+Bg')
            plt.title("Source and background, pixel ["
                      +str(self.plotpix[0])+","+str(self.plotpix[1])+"]")
            plt.xlabel("Velocity [m/s]")
            plt.ylabel("Flux [e-/sec]")
            plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
            plt.ylim(bottom=0)
            plt.show()

        # Decide on the DIT to use, full well = 100e3, we fill to 80%
        #dit = 80e3 * u.photon / np.max(ph_cube)
        #print("Peak", np.max(ph_cube), ", using DIT", dit)

        dit = exptime * u.s / ndit
        print("Exptime", exptime, "s, NDIT =", ndit, ", using DIT", dit)

        ph_cube *= dit
        bg_cube *= dit

        print("Peak in one DIT", np.max(ph_cube))
        sat = len(np.where(ph_cube.value > 100e3)[0])
        if sat > 0:
            warn = "\n!!!WARNING: "+str(sat)+" PIXELS ARE ABOVE THE FULL-WELL CAPACITY OF 100000 PHOTONS!!!"
            warnings.warn(warn)

        if np.max(ph_cube.value) < 10e3:
            warnings.warn("\nWARNING: brightest pixel has <10% of full well capacity. Your data will be noisy!")

        targ_noise = np.sqrt(ph_cube*u.electron + (70 * u.electron)**2)	# RON = 70e/pix/read
        back_noise = np.sqrt(bg_cube*u.electron + (70 * u.electron)**2)	# RON = 70e/pix/read

        #ndit = np.round(integration_time / dit)
        #print("Total integration time", integration_time, "=> NDIT =", ndit)

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

        key = 'HIERARCH '+self.cmds.cmds.popitem(last=False)[0]
        header.set('EXPTIME', exptime, "[s] Total exposure time = DIT*NDIT", before=key)
        header.set('DIT', dit.value, "[s] Detector integration time = EXPTIME/NDIT", before=key)
        header.set('NDIT', ndit, "Number of integrations = EXPTIME/DIT", before=key)
        self.add_cmds_to_header(header)

        if write_src_w_bg is not None:
            print("Writing source+background to \"", write_src_w_bg, '"')
            hdu = fits.PrimaryHDU(ph_cube.value, header=header)
            hdu.writeto(write_src_w_bg, overwrite=True)

        if write_background is not None:
            print("Writing background to \"", write_background, '"')
            hdu = fits.PrimaryHDU(bg_cube.value, header=header)
            hdu.writeto(write_background, overwrite=True)

        ph_cube -= bg_cube
        hdu = fits.PrimaryHDU(ph_cube.value, header=header)
        return hdu


    #############################################################################

    def calibrate_flux(self, hdu):
        '''
        do flux calibration by undoing what the simulator did so far (as much as possible)
        input is the result of compute_snr()
        '''
        data = hdu.data * u.electron	# per dit, pixel, spectral channel, and M1-area

        mirr_list = self.cmds.mirrors_telescope
        mirr_area = np.pi / 4 * np.sum(mirr_list["Outer"]**2 - \
                                       mirr_list["Inner"]**2) * u.m**2

        data = data / (hdu.header['EXPTIME'] * u.s
                       * (np.mean(self.wavelen)*u.um * (1.5 *u.km/u.s) / const.c).to(u.um)
                       * mirr_area)
        # e-/s/um/m2

        # wavelengths of data cube
        det_wavelen = (self.det_velocities
                       * u.m/u.s).to(u.um, equivalencies=u.doppler_optical(self.restcoo))

        # interpolate transmission onto wavelength-grid of detector:
        trans = np.interp(det_wavelen, self.wavelen, self.transmission)

        data /= trans[:, np.newaxis, np.newaxis]

        data = (data * u.photon/u.electron).to(u.Jy,
                                               equivalencies=u.spectral_density(self.restcoo))

        data = data / (self.det_pixscale/1000. * u.arcsec)**2
        # Jy/arcsec2

        hdu.data = data.value
        hdu.header['BUNIT'] = ('JY/ARCSEC2', 'Jansky per arcsec**2')

        return hdu


    #############################################################################

    def save_cube(self, outname):
        '''write the data cube to a fits-file'''

        print("Writing data cube to",outname)
        hdu = fits.PrimaryHDU(self.target_cube, header=self.target_hdr)
        hdu.writeto(outname, overwrite=True)


    #############################################################################

    def simulate(self, conditions, psf_name, exptime, ndit, plot=False):
        '''run a LMS simulation'''

        self.transmission_emission(conditions=conditions, plot=plot)
        self.convolve_psf(psf_name, plot=plot)
        self.convolve_lsf(plot=plot)
        self.scale_to_detector()
        return self.compute_snr(exptime=exptime, ndit=ndit, plot=plot)


#############################################################################
