import warnings
import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import sigma_clipped_stats
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.stats import sigma_clipped_stats

from photutils import Background2D, SigmaClip, MedianBackground
from photutils import DAOStarFinder
from photutils.psf import DAOPhotPSFPhotometry, IntegratedGaussianPRF, \
                          DAOGroup, BasicPSFPhotometry
from photutils.background import MMMBackground


import aplpy
import simcado as sim


class Stamp(object):

    def __init__(self, image, image_id=None, fwhm=None, **kwargs):

        self.params = {"id"     : image_id,
                       "mag_zero_point"    : 0,
                       "aperture_radius"   : 3.,
                       "sky_radius_inner"  : None,
                       "sky_radius_outer"  : None,
                       "hot_pixel_threshold" : 3000,
                       "photometry_mode"   : "psf"
                      }
        self.params.update(kwargs)

        self.image = remove_hot_pixels(image, threshold=self.params["hot_pixel_threshold"])
        self.results = None
        #try:
        #   self.bg_mean, self.bg_median, self.bg_std = \
        #                    sigma_clipped_stats(self.image, sigma=3., iters=3)
        #except:
        #    self.bg_mean, self.bg_median, self.bg_std = 0, 0, 0

        try:
            height, x, y, width_x, width_y = self.get_moments()
        except:
            height, x, y, width_x, width_y = [0]*5

        
        self.x,  self.y  = x, y
        self.dx, self.dy = width_x, width_y
        if fwhm is None:
            self.fwhm = self.get_fwhm()
        else:
            self.fwhm = fwhm
        
        self.mag, self.snr, self.flux, self.noise, self.bg_std, self.bg_median = \
                            self.get_photometry(self.params["aperture_radius"],
                                                self.fwhm,
                                                self.params["sky_radius_inner"],
                                                self.params["sky_radius_outer"],
                                                self.params["mag_zero_point"],
                                                self.params["photometry_mode"])
        
        self.peak = self._get_peak()
        
        

    def get_photometry(self, aperture_radius=3, fwhm=None,
                       sky_radius_inner=None, sky_radius_outer=None,
                       mag_zero_point=0, mode="basic"):


        if sky_radius_outer is None:
            sky_radius_outer = np.min(self.image.shape) // 2
        if sky_radius_inner is None:
            sky_radius_inner = sky_radius_outer - 3

        x, y = np.array(self.image.shape) // 2

        r  = aperture_radius
        ro = sky_radius_outer
        dw = sky_radius_outer - sky_radius_inner

        bg = np.copy(self.image[y-ro:y+ro, x-ro:x+ro])
        bg[dw:-dw, dw:-dw] = 0
        bg_median = np.median(bg[bg!=0])
        bg_std = np.std(bg[bg!=0])
        noise = bg_std * np.sqrt(np.sum(bg != 0))

        im = np.copy(self.image[y-r:y+r, x-r:x+r])
        
        if "circ" in mode.lower():
            pass
        
        elif "basic" in mode.lower():
            flux = np.sum(im - bg_median)
            self.results = None
            
            
        elif "psf" in mode.lower():    
            
            if fwhm is None:
                warnings.warn("``fwhm`` must be given for PSF photometry")
                flux = 0 
            
            try:
                x,y  = self._find_center(self.image, fwhm)   
                flux = self._basic_psf_flux(self.image, fwhm, x=x, y=y)
            ##### Really bad form!!! Keep this in mind =)
                self.x, self.y = x, y
            except:
                flux = 0
            

            
        elif "dao" in mode.lower():

            if fwhm is None:
                warnings.warn("``fwhm`` must be given for PSF photometry")
                flux = 0 

            sigma = fwhm / 2.35
            prf = IntegratedGaussianPRF(sigma)
            fitshape = im.shape[0] if im.shape[0] % 2 == 1 else im.shape[0] - 1

            daophot = DAOPhotPSFPhotometry(crit_separation=sigma, 
                                           threshold=np.median(im), 
                                           fwhm=fwhm, psf_model=prf, 
                                           fitshape=fitshape)
            
            #try:
            results = daophot(im)

            width, height = im.shape
            dist_from_centre = np.sqrt((results["x_fit"] - width  / 2)**2 + \
                                       (results["y_fit"] - height / 2)**2)
        
            i = np.argmin(dist_from_centre)
            x, y  = results["x_fit"][i], results["y_fit"][i], 
            flux  = results["flux_fit"][i]
            
            ##### Really bad form!!! Keep this in mind =)
            self.x, self.y = x, y
            self.results = results
            
            #except:
            #    self.results = None
             #   flux = 0
                
                        
        snr = flux / noise
        mag = -2.5 * np.log10(flux) + mag_zero_point

        return mag, snr, flux, noise, bg_std, bg_median


    def get_moments(self):
    
        return gaussian_moments(self.image)

    def get_fwhm(self):
        
        fwhm = 2.35 * 0.5 * (self.dx + self.dy)
        if np.isnan(fwhm):
            fwhm = 1
        elif fwhm < 1:
            fwhm = 1
        elif fwhm > 50:
            fwhm = 50
        
        return fwhm

    def _get_peak(self, n=3):
        
        xc, yc = int(self.x), int(self.y)
        try:
            peak = np.max(self.image[yc-n:yc+n, xc-n:xc+n])
        except:
            peak = 0
            
        return peak
        
        
    def _find_center(self, image, fwhm):
    
        mean, median, std = sigma_clipped_stats(image, sigma=3.0, iters=5)
        im = image - median

        daofind = DAOStarFinder(fwhm=fwhm, threshold=5*std)
        results = daofind(im)

        w, h = im.shape
        if len(results) > 0:
            dist_from_centre = np.sqrt((results["xcentroid"] - w / 2)**2 + (results["ycentroid"] - h / 2)**2)
            i = np.argmin(dist_from_centre)
            x, y = results[i]["xcentroid"], results[i]["ycentroid"]
        else:
            x, y = w/2, h/2
        return x, y 


    def _basic_psf_flux(self, image, fwhm, x=None, y=None, return_residual_image=False):
       
        w,h = image.shape
        if x is None:
            x = w / 2
        if y is None:
            y = h / 2

        wfit = w if w % 2 == 1 else w - 1
        hfit = h if h % 2 == 1 else h - 1
        fitshape = (wfit, hfit)
        
        daogroup = DAOGroup(2.0*fwhm)
        psf_model = IntegratedGaussianPRF(sigma=fwhm/2.35)

        photometry = BasicPSFPhotometry(group_maker=daogroup,
                                        bkg_estimator=MMMBackground(),
                                        psf_model=psf_model,
                                        fitshape=fitshape)

        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True
        pos = Table(names=['x_0', 'y_0'], data=[[x],[y]])

        result = photometry(image=image, positions=pos)
        flux = result["flux_fit"].data[0]

        self.results = result
        
        if return_residual_image:
            return flux, photometry.get_residual_image()
        else:
            return flux
        


class PostageStamps(object):

    def __init__(self, image, x=None, y=None, name=None, **kwargs):


        self.params = {"bg_tile_size"      : 32,
                       "fwhm"              : 5.,
                       "stamp_width"       : 32,
                       "stamp_height"      : None,
                       "mag_zero_point"    : 0,
                       "aperture_radius"   : 3.,
                       "sky_radius_inner"  : None,
                       "sky_radius_outer"  : None,
                       "threshold"         : None,
                       "photometry_mode"   : "psf",
                      }

        self.params.update(kwargs)


        self.image_bg = self._get_background(image,
                                             tile_size=self.params["bg_tile_size"])
        self.image = image - self.image_bg

        if x is None and y is None:
            self.x, self.y = self.find_sources(fwhm=self.params["fwhm"],
                                               threshold=self.params["threshold"])
        elif x is not None and y is not None:
            self.x, self.y = x, y
        else:
            raise ValueError("x and y need to be both None or equal length arrays")

        self.stamps = self.get_stamps(self.x, self.y,
                                      w=self.params["stamp_width"],
                                      h=self.params["stamp_height"])

        self._get_photometry()


    def find_sources(self, fwhm=5., threshold=None):

        mean, median, std = sigma_clipped_stats(self.image, sigma=3., iters=3)

        if threshold is None:
            threshold=5.*std

        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
        sources = daofind(self.image - median)
        self.DAOsources = sources

        return sources["xcentroid"], sources["ycentroid"]


    def get_stamps(self, x, y, w=16, h=None):

        if h is None:
            h = w

        x0, x1 = x - w//2, x - w//2 + w
        y0, y1 = y - h//2, y - h//2 + h
        ims = [self.image[yy0:yy1, xx0:xx1] for yy0, yy1, xx0, xx1 in zip(y0, y1, x0, x1)]

        stamps = [Stamp(im, **self.params) for im in ims]

        return stamps


    def _get_background(self, image, tile_size=32):

        sigma_clip = SigmaClip(sigma=3., iters=3)
        bkg_estimator = MedianBackground()
        bkg = Background2D(image, tile_size, filter_size=3,
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        return bkg.background


    def _get_photometry(self):

        self.mags      = np.array([stamp.mag   for stamp in self.stamps])
        self.snrs      = np.array([stamp.snr   for stamp in self.stamps])
        self.fluxes    = np.array([stamp.flux  for stamp in self.stamps])
        self.noises    = np.array([stamp.noise for stamp in self.stamps])
        self.fwhms     = np.array([stamp.fwhm  for stamp in self.stamps])
        self.peaks     = np.array([stamp.peak  for stamp in self.stamps])
        self.bg_stds   = np.array([stamp.bg_std     for stamp in self.stamps])
        self.bg_medians = np.array([stamp.bg_median for stamp in self.stamps])
        self.results   = [stamp.results for stamp in self.stamps]

    def plot_stamps(self, n, n_wide=5, colorbar=False, vmin=None, vmax=None, norm=None):

        if isinstance(n, str) and n == "all":
            n = range(len(self.stamps))

        if np.isscalar(n):
            n = range(n)

        w = n_wide
        l = len(n)
        h = l // w + 1

        for i, plot_i in zip(n, range(len(n))):
            plt.subplot(h, w, plot_i+1)
            plt.imshow(self.stamps[i].image, norm=norm, vmin=vmin, vmax=vmax)

            if colorbar:
                plt.colorbar()


def gaussian_moments(data):
    """
    Returns (height, x, y, width_x, width_y) for a 2D Gaussian

    Taken from:
        http://scipy-cookbook.readthedocs.io/items/FittingData.html

    Parameters
    ----------
    data : np.array
        2D Gaussian

    Returns
    -------
    height, x, y, width_x, width_y

    """
    height = data.max()

    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total

    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())

    return height, x, y, width_x, width_y
    
    
def remove_hot_pixels(image, threshold=5000):    
    """
    
    Parameters
    ----------
    image : np.ndarray
        The image, in 2D format
    threshold : float, int
        The difference between any two neighbbouring pixels. 

    
    Returns
    -------
    a : np.array
        The input image with the hot pixels set to the image median
    
    """
    a = np.copy(image)
    b = a[:-1,:] - a[1:,:]
    c = b[1:,:] - b[:-1,:]

    x, y = np.where(c > threshold)
    a[x+1, y] = np.median(a)
    
    return a
    
    
def plot_catalogue_on_image(catalogue, hdu, hdu_ext, dx=0, dy=0, cat_ra_name="RA", cat_dec_name="DE", cat_mag_name="J"):
    """
    Overplots a scatter of circles on the FITS image
    
    Parameters
    ----------       
    catalogue : str, astropy.Table
        A catalogue of stars - the same as used in mimic_image()
                
    hdu : str, astropy.HDUList
        The original FITS object- either a filename or an astropy.HDUList object
        
    hdu_ext : int
        The extension in the original FITS file which should be simulated
    
    dx, dy : float, int
        Offset in pixels between the image and the catalogue coordinates
        
    cat_ra_name, cat_dec_name, cat_filter_name : str
        Names of the columns in the catalogue for RA, Dec and the magnitude values
        
    
    Examples
    --------
    ::
    
        >>> plot_catalogue_on_image(cat, hdu_real, 3)
        
    """

    n = 0
    #w, h = np.shape(hdu[hdu_ext].data)
    #mask = (src.x_pix > n) * (src.x_pix < w-n) * (src.y_pix > n) * (src.y_pix < h-n)

    cat = catalogue
    xw  = cat[cat_ra_name]
    yw  = cat[cat_dec_name]
    mag = cat[cat_mag_name]

    fig = plt.figure(figsize=(10,10))

    apl_fig = aplpy.FITSFigure(hdu[hdu_ext], figure=fig)
    xpr, ypr = apl_fig.world2pixel(xw, yw)

    xpr += dx
    ypr += dy

    im_real = np.copy(hdu[hdu_ext].data)
    im_real -= np.median(im_real)

    plt.scatter(xpr, ypr, s=(19-mag)*10, alpha=0.5)

    plt.imshow(im_real, origin="lower", cmap="hot", vmax=5E2, vmin=1)
    plt.colorbar()

    plt.title(hdu[0].header["DATE-OBS"])
    plt.xlim(-50, 2100); plt.ylim(-50, 2100)        
    
    
    
def mimic_image(hdu, catalogue, cmds=None, hdu_ext=0, sim_chip_n=0, return_stamps=False, 
                cat_ra_name="RA", cat_dec_name="DE", cat_filter_name="J", **kwargs):
    """
    Create a SimCADO image to mimic a real FITS file
    
    Parameters
    ----------
    hdu : str, astropy.HDUList
        The original FITS object- either a filename or an astropy.HDUList object
    
    catalogue : str, astropy.Table
        A catalogue of stars - either a filename or an astropy.Table object
    
    cmds : str, simcado.UserCommands
        Commands by which to simulate the image - a filename to a .config file or a UserCommands object
    
    hdu_ext : int
        The extension in the original FITS file which should be simulated
    
    sim_chip_n : int
        Which chip in the FPA_LAYOUT to simulate. Passed to apply_optical_train() and read_out()
    
    return_stamps : bool
        If True, returns two PostageStamps object for the original HDU and the generated HDU
   
    cat_ra_name, cat_dec_name, cat_filter_name : str
        The names of the column in catalogue which point to the RA, Dec and filter magnitudes for the stars
    
        
    Optional Parameters
    -------------------
    As passed to a PostageStamps object
    
    "dx"           : 0,
    "dy "          : 0,
    "bg_tile_size" : 24, 
    "stamp_width"  : 24, 
    "hot_pixel_threshold" : 3000 
    
    
    Returns
    -------
    hdu_sim, src [, post_real, post_sim]
        If return_stamps is True, post_real and post_sim are returned
        
        
    Examples
    --------
    ::
    
        >>> hdu_real = fits.open("HAWKI.2008-11-05T04_08_55.552.fits")
        >>> cat = ascii.read("ngc362_cohen.dat")    
        >>> 
        >>> dx, dy = -5, 15
        >>> 
        >>> cmd = sim.UserCommands("hawki_ngc362.config")
        >>> cmd["INST_FILTER_TC"] = "J"
        >>> cmd["OBS_EXPTIME"] = 10
        >>> 
        >>> out = mimic_image(hdu=hdu_real, catalogue=cat, cmds=cmd, hdu_ext=3, 
        ...                   sim_chip_n=3, return_stamps=True, 
        ...                   dx=dx, dy=dy)
        >>> hdu_sim, src, post_real, post_sim = out
        >>> len(out)
        4
        
    """
    
    if isinstance(hdu, str) and os.path.exists(hdu):
        hdu = fits.open(hdu)[hdu_ext]
    elif isinstance(hdu, fits.HDUList):
        hdu = hdu[hdu_ext]
    else:
        raise ValueError("hdu must be a filename or an astropy HDU object: "+type(hdu))
    
    if isinstance(catalogue, str) and os.path.exists(catalogue):
        cat = ascii.read(catalogue)
    elif isinstance(catalogue, Table):
        cat = catalogue
    else:
        raise ValueError("catalogue must be a filename or an astropy.Table object: "+type(catalogue))
    
    if isinstance(cmds, str) and os.path.exists(cmds):
        cmds = sim.UserCommands(cmds)
    elif isinstance(cmds, sim.UserCommands):
        pass
    else:
        raise ValueError("cmds must be a filename or an simcado.UserCommands object: "+type(cmds))
    
    
    fig = plt.figure(figsize=(0.1,0.1))
    apl_fig = aplpy.FITSFigure(hdu, figure=fig)

    # get the RA DEC position of the centre of the HAWKI FoV
    xc, yc = hdu.header["CRPIX1"], hdu.header["CRPIX2"]
    ra_cen, dec_cen = apl_fig.pixel2world(xc, yc)

    # get the x,y positions in arcsec from the HAWKI FoV centre
    y = (cat[cat_dec_name] - dec_cen) * 3600
    x = -(cat[cat_ra_name] - ra_cen) * 3600 * np.cos(cat[cat_dec_name]/57.3)
    mag = cat[cat_filter_name]

    # make a Source object with the x,y positions in arcsec from the HAWKI FoV centre
    src = sim.source.stars(mags=mag, filter_name=cat_filter_name, x=x ,y=y)

    opt = sim.OpticalTrain(cmds)
    fpa = sim.Detector(cmds, small_fov=False)

    print(sim_chip_n)
    
    src.apply_optical_train(opt, fpa, chips=sim_chip_n)
    hdu_sim = fpa.read_out(chips=sim_chip_n)

    ## Get the Postage Stamps
    if return_stamps:
        
        params = {"dx"           : 0,
                  "dy "          : 0,
                  "bg_tile_size" : 24, 
                  "stamp_width"  : 24, 
                  "hot_pixel_threshold" : 3000 }
        params.update(**kwargs)

        w, h = hdu_sim[0].data.shape
        mask = (src.x_pix > 0) * (src.x_pix < w) * (src.y_pix > 0) * (src.y_pix < h)

        xw = cat[cat_ra_name][mask]
        yw = cat[cat_dec_name][mask]
        mag = cat[cat_filter_name][mask]

        # get the x,y pixel positions of the stars in the simulated image
        xps = src.x_pix[mask]
        yps = src.y_pix[mask]

        # get the x,y pixel positions of the stars in the real image, include offset if needed
        xpr, ypr = apl_fig.world2pixel(xw, yw)
        xpr += params["dx"]
        ypr += params["dy"]

        # get the images from the FITS objects
        im_sim = np.copy(hdu_sim[0].data)
        im_sim -= np.median(im_sim)
        post_sim = PostageStamps(im_sim, x=xps, y=yps, **params) 

        im_real = np.copy(hdu.data)
        im_real -= np.median(im_real)
        post_real = PostageStamps(im_real, x=xpr, y=ypr, **params) 

        return hdu_sim, src, post_real, post_sim
    
    else: 
        return hdu_sim, src



def plot_compare_stamps(post_real, post_sim, hdu_real, hdu_sim, 
                        plot_limits=[[1E1, 1E6],  [1E2, 1E7],[0, 30],
                                     [-100, 1000],[1E1, 1E4],[3E0, 1E2]],
                        fwhm_limiting_mag=13.5, zero_point_mag=26,
                        hdr_seeing=None, clr="r", marker="+"):
    """
    Plot a comparison of two PostageStamp objects and two HDULists
    
    Parameters
    ----------
    post_real : PostageStamps object
        Made from the real FITS image. Third output from mimic_image()
        
    post_sim : PostageStamps object
        Made from the simulated FITS image. Fourth output from mimic_image()
    
    hdu_real : astropy.HDUList
        The original FITS file
    
    hdu_sim : astropy.HDUList
        The simulated FITS file. First output from mimic_image()
    
    plot_limits : list
        As the name says
    
    fwhm_limiting_mag, zero_point_mag : float
        Magnitudes for scaling the plots
        
    hdr_seeing : int, float
        The Seeing FWHM from the FITS header
    
    
    Examples
    --------
    ::
    
        >>> plot_compare_stamps(post_real, post_sim, hdu_real[3], hdu_sim[0])

    """
    
    mag = post_sim.mags + zero_point_mag

    plt.figure(figsize=(12,10))
    
    ###############################################
    # 1 - Peak flux value

    plt.axes((0,0.65,0.45,0.35))
    qmin, qmax = plot_limits[0]

    plt.plot(post_real.peaks, post_sim.peaks, clr+marker)
    plt.plot([qmin, qmax], [qmin, qmax], "k--")

    plt.loglog()
    plt.xlim(qmin,qmax); plt.ylim(qmin,qmax)
    #plt.xlabel("Real Flux"); 
    plt.ylabel("SimCADO Flux")
    plt.title("Peak flux value")
    
    plt.xticks([], [])

    ###############################################
    # 2 - Integrated flux inside aperture

    plt.axes((0.55,0.65,0.45,0.35))
    qmin, qmax = plot_limits[1]

    plt.plot(post_real.fluxes, post_sim.fluxes, clr+marker)
    plt.plot([qmin, qmax], [qmin, qmax], "k--")

    plt.loglog()
    plt.xlim(qmin,qmax); plt.ylim(qmin,qmax)
    #plt.xlabel("Real Flux"); 
    plt.ylabel("SimCADO Flux")
    plt.title("Integrated flux inside aperture")

    plt.xticks([], [])
    
    ###############################################
    # 3 - Residuals for peak flux

    plt.axes((0.,0.5,0.45,0.15))
    qmin, qmax = plot_limits[0]

    a = post_real.peaks
    b = post_sim.peaks
    
    c = (a-b)/a
    
    plt.plot(a, c, clr+marker)
    plt.plot([qmin, qmax], [0, 0], "k--")
    plt.plot([qmin, qmax], [np.median(c), np.median(c)], "b")
    #plt.text(qmin, 2, np.round(np.median(c), 3), np.round(np.std(c), 3))
    
    plt.semilogx()
    plt.xlim(qmin,qmax)
    plt.ylim(-3,2.9)
    plt.xlabel("Real Flux")
    plt.ylabel("Residual factor")
    #plt.title("Integrated flux inside aperture")
    
    
    ###############################################
    # 4 - Residuals for Integrated flux

    plt.axes((0.55,0.5,0.45,0.15))
    qmin, qmax = plot_limits[1]

    a = post_real.fluxes
    b = post_sim.fluxes
    
    c = (a-b)/a
    
    plt.plot(a, c, clr+marker)
    plt.plot([qmin, qmax], [0, 0], "k--")
    plt.plot([qmin, qmax], [np.median(c), np.median(c)], "b")
    #plt.text(qmin, 2, np.round(np.median(c), 3), np.round(np.std(c), 3))

    plt.semilogx()
    plt.xlim(qmin,qmax)
    plt.ylim(-3,2.9)
    plt.xlabel("Real Flux")
    plt.ylabel("Residual factor")
    #plt.title("Integrated flux inside aperture")    
    
    
    ###############################################
    # 5 - FWHM of sources in pixels

    plt.axes((0.0,0.0,0.45,0.4))
    qmin, qmax = plot_limits[2]
    mask = mag < fwhm_limiting_mag

    plt.scatter(post_real.fwhms[mask], post_sim.fwhms[mask], c=clr, s=100*(fwhm_limiting_mag-mag[mask])**2, alpha=0.5)
    plt.plot([qmin, qmax], [qmin, qmax], "k--")
        
    fwhms = np.array(post_real.fwhms[mask])
    mask = (fwhms > 1) * (fwhms < 50)
    # np.invert(np.isnan(fwhms))
    fwhms = fwhms[mask]
    av = np.median(fwhms)
    plt.plot([av,av], [qmin, qmax], "k:")
    
    fwhms = np.array(post_sim.fwhms[mask])
    #fwhms = fwhms[np.invert(np.isnan(fwhms))]
    av = np.median(fwhms)
    plt.plot([qmin, qmax], [av,av], "k:")

    if hdr_seeing is not None and isinstance(hdr_seeing, (int, float)):
        plt.scatter(hdr_seeing, av, c="g", s=20, marker="^")
    
    plt.xlim(qmin,qmax); plt.ylim(qmin,qmax)
    plt.xlabel("Real FWHM"); plt.ylabel("SimCADO FWHM")
    plt.title("FWHM of sources in pixels")

    ###############################################
    # 6 - Histogram of pixel values

    plt.axes((0.55,0.0,0.45,0.4))
    qmin, qmax = plot_limits[3]

    y, x = np.histogram(hdu_real.data.flatten(), bins=np.logspace(2, 5, 100))
    plt.plot(x[1:], y, "b", label="Real image")

    y, x = np.histogram(hdu_sim.data.flatten(), bins=np.logspace(2, 5, 100))
    plt.plot(x[1:], y, "r", label="Simulated image")

    plt.loglog()
    plt.legend(loc=2)
    plt.xlabel("Pixel Value"); plt.ylabel("Number of pixels")
    plt.title("Histogram of pixel values")

    
def things():
    ###############################################
    # Std in BG for stamps
    plt.subplot(235)
    qmin, qmax = plot_limits[4]

    plt.plot(post_real.bg_stds, post_sim.bg_stds, clr+marker)
    plt.plot([qmin, qmax], [qmin, qmax], "k--")

    plt.loglog()
    plt.xlim(qmin,qmax); plt.ylim(qmin,qmax)
    plt.xlabel("Real Flux"); plt.ylabel("SimCADO Flux")
    plt.title("Standard deviation in background flux")

    ###############################################
    # SNR for point sources

    plt.subplot(236)
    qmin, qmax = plot_limits[5]

    plt.plot(post_real.snrs, post_sim.snrs, clr+marker)
    plt.plot([qmin, qmax], [qmin, qmax], "k--")
    plt.plot([5, 5], [qmin, qmax], "k:")
    plt.plot([qmin, qmax], [5, 5], "k:")

    plt.loglog()
    plt.xlim(qmin,qmax); plt.ylim(qmin,qmax)
    plt.xlabel("Real SNR"); plt.ylabel("SimCADO SNR")
    plt.title("SNR for point sources")        