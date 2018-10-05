import numpy as np

import scipy.optimize as opt
from scipy.stats import linregress
from scipy.stats.mstats import theilslopes
from scipy.signal import fftconvolve

from astropy.io import fits, ascii
from astropy.stats import sigma_clipped_stats
from astropy.table import vstack, hstack, Table

try:
    from photutils import DAOStarFinder
except:
    pass

import simmetis as sim


def make_uniform_dense_core_image(mass, dist_mod, side_width=1024, filename=None,
                                  n_stack=1, pix_res=0.004, **kwargs):
    """
    Makes an image with a uniformly random distribution of stars from an IMF

    Parameter
    ---------
    mass : float
        [Msun]

    dist_mod : float
        [mag]

    side_width : int
        [pixel]

    filename : str

    n_stack : int

    pix_res : float
        [arcsec/pixel] Default is MICADO Wide: 4mas/px


    Optional Parameters
    -------------------
    SCOPE_PSF_FILE
    OBS_REMOVE_CONST_BG
    OBS_EXPTIME

    """

    params = {"SCOPE_PSF_FILE" : None,
              "OBS_REMOVE_CONST_BG" : "yes",
              "OBS_EXPTIME" : 60}
    params.update(kwargs)

    density = mass / (side_width * pix_res)**2

    masses = imf_population(mass=mass)
    mags = abs_mag_from_mass(masses) + dist_mod
    n_stars = len(masses)
    x, y = np.random.randint(-side_width//2, side_width, (2, n_stars))

    src = sim.source.stars(mags=mags, x=x, y=y, pix_unit="pixel")

    hdu, (cmd, opt, fpa) = sim.run(src, return_internals=True, **kwargs)

    for i in range(1, n_stack):
        hdu[0].data += fpa.read_out()[0].data

    if filename is None:
        filename = "rho="+str(int(density))+".tbl"

    tbl = Table(data=[x, y, masses, mags], names=["x", "y", "masses", "mags"])
    tbl.write(filename, format="fits",  overwrite=True)

    f = fits.open(filename)
    hdu.append(f[1])
    hdu.writeto(filename.replace(".tbl", ".fits"), clobber=True)
    f.close()

    return hdu


def binned_clipped_stats(x, y, bins, **kwargs):
    """
    Sigma clipped stats for binned data
    """

    m_stats = []
    for i in range(len(bins)-1):
        xmin, xmax = bins[i], bins[i+1]
        mask = (x > xmin) * (x <= xmax)
        m_stats += [sigma_clipped_stats(y[mask], **kwargs)]

    m_stats = np.array(m_stats)

    tbl_1 = Table(data=[bins[:-1], bins[1:], 0.5*(bins[:-1]+bins[1:]) ],
                  names=["x_min", "x_max", "x_center"])
    tbl_2 = Table(data=m_stats,
                  names=["mean", "median", "std"])

    tbl_stats = hstack([tbl_1, tbl_2])

    return tbl_stats



def power_law(alphas, lims, ddex=0.1, scale_factors=None):
    """
    Returns

    Parameters
    ----------
    alphas : list
        The power law indicies

    lims : list of limit pairs
        (N, 2) sized list for the minimum and maximum x-axis limits for each
        segment of the power law. The second value of the n-1 pair must be equal
        to the first entry of n pair.
        E.g. [[0, 1], [1, 10]]

    ddex : float
        The resolution of the returned  curve_fit

    scale_factors : list
        A list of scaling factors for the parts of various parts of the curve
        len(scale_factors) must be equal to len(alphas)


    Returns
    -------
    x, y : np.ndarray
        x and y coordinates for the combined power law curve


    """

    if scale_factors is None:
        scale_factors = [1]*len(alphas)

    yy = []
    xx = []
    ff = []
    for a, lim, sf in zip(alphas, lims, scale_factors):
        n_bins = np.int((np.log10(lim[1]) - np.log10(lim[0])) / ddex) + 1

        xe = np.logspace(np.log10(lim[0]), np.log10(lim[1]), n_bins)
        #me = np.linspace(lim[0], lim[1], n_bins)
        xc = (xe[1:] + xe[:-1]) * 0.5

        yc = xc**a
        yy += [yc]
        xx += [xc]
        ff += [sf]*len(xc)

    for i in range(1, len(yy)):
        f = yy[i-1][-1] / yy[i][0]
        yy[i] *= f

    x = np.concatenate(xx)
    y = np.concatenate(yy)
    f = np.array(ff)

    y *= f
    y /= y.sum()

    return x, y


def imf_population(mass=1000, ddex=0.01, alphas=[0.3, 1.3, 2.3],
                   lims=[[1E-3, 0.08], [0.08, 0.5], [0.5, 200]],
                   scale_factors=[0.3, 1, 1]):
    """
    Returns a random list of masses based on a broken power law distribution

    Parameters
    ----------
    mass : float
        Mass of the cluster. Default is 1000 Msun

    ddex : float
        Bin sizes in logspace for sampling the distribution. Default is 0.01

    alphas : list
        Power law indicies for each section of the broken power law distribution
        Default is for a Kroup IMF

    lims : list
        The x-axis limits of each of the sections of the broken power law
        Default is for a Kroupa IMF

    scale_factors : list
        To take into account the Brown Dwarf desert. Defaults are [0.3, 1, 1]


    Returns
    -------
    masses : list
        A list of masses sampled from the broken power law distribtion

    """


    alphas = 1 - np.array(alphas)
    mcs, ns = power_law(alphas, lims, ddex, scale_factors)

    ns /= np.sum(ns)

    expectation_mass = np.sum(mcs*ns) / np.sum(ns)

    masses = np.random.choice(a=mcs, size=int(1.3 * mass / expectation_mass), p=ns)
    w = np.cumsum(masses)
    if w[-1] > mass:
        i = np.where(w > mass)[0][0]
    else:
        i = len(w)
    masses = masses[:i]

    print("Cluster mass: ", w[i-1])

    return masses


def luminosity_from_mass(mass, cat="../MS-stars.txt", col_mag="M_Ks", col_mass="Msun",
                         flux_0=1., dist_mod=0.):
    """
    Gets stellar luminosities based on the mass

    Parameters
    ---------
    mass : float, array
        The mass of the star(s)

    cat : str
        The catalogue containing the magnitude-mass relationship

    col_mag : str
        The name of the column in ``cat`` containing the magnitude values

    col_mass : str
        The name of the column in ``cat`` containing the mass values

    flux_0 : float
        Flux value for a mag=0 star. Used to scale the ``luminosity`` values
        from the catalogue

    dist_mod : float
        In case there is a magnitude offset, this can be included here


    Returns
    -------
    luminosity : float, array
        Luminosity values for all ``mass`` values relative to a mag=0 star


    """

    if isinstance(cat, str):
        MScat = ascii.read(cat)
    elif isinstance(cat, Table):
        MScat = cat

    cat_mass = MScat[col_mass].data
    cat_Ks = MScat[col_mag].data

    ks = np.interp(mass, cat_mass[::-1], cat_Ks[::-1])
    ks += dist_mod
    fluxes = flux_0 * 10**(-0.4*ks)

    return fluxes


def abs_mag_from_mass(mass, cat="../MS-stars.txt", col_mag="M_Ks", col_mass="Msun"):
    """
    Gets stellar luminosities based on the mass

    Parameters
    ---------
    mass : float, array
        The mass of the star(s)

    cat : str
        The catalogue containing the magnitude-mass relationship

    col_mag : str
        The name of the column in ``cat`` containing the magnitude values

    col_mass : str
        The name of the column in ``cat`` containing the mass values

    flux_0 : float
        Flux value for a mag=0 star. Used to scale the ``luminosity`` values
        from the catalogue

    dist_mod : float
        In case there is a magnitude offset, this can be included here


    Returns
    -------
    luminosity : float, array
        Luminosity values for all ``mass`` values relative to a mag=0 star


    """

    if isinstance(cat, str):
        MScat = ascii.read(cat)
    elif isinstance(cat, Table):
        MScat = cat

    cat_mass = MScat[col_mass].data
    cat_Ks = MScat[col_mag].data

    mags = np.interp(mass, cat_mass[::-1], cat_Ks[::-1])

    return mags


def mass_from_luminosity(luminosity, cat="../MS-stars.txt", col_mag="M_Ks",
                         col_mass="Msun", flux_0=1., dist_mod=0.):
    """
    Gets a stellar masses based on luminosities

    Parameters
    ---------
    luminosity : float, array
        The flux of the star compared to a mag=0 star. Flux of mag=0 star can be
        specified with the parameter flux_0

    cat : str
        The catalogue containing the magnitude-mass relationship

    col_mag : str
        The name of the column in ``cat`` containing the magnitude values

    col_mass : str
        The name of the column in ``cat`` containing the mass values

    flux_0 : float
        Flux value for a mag=0 star. Used to normalise the ``luminosity`` values
        to match the catalogue

    dist_mod : float
        In case there is a magnitude offset, this can be included here


    Returns
    -------
    masses : float, array
        Mass values for all ``luminosity`` values


    """

    if isinstance(cat, str):
        MScat = ascii.read(cat)
    elif isinstance(cat, Table):
        MScat = cat

    cat_mass = MScat[col_mass].data
    cat_Ks = MScat[col_mag].data

    ks = -2.5*np.log10(luminosity / flux_0)
    ks -= dist_mod

    masses = np.interp(ks, cat_Ks, cat_mass)

    return masses


def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    """
    Used for fitting a 2D gaussian

    """

    (x, y) = xy
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()


def get_flux(im, plate_scale, radius=9, xp=None, yp=None):
    """
    Return flux based on different photometric methods

    Parameters
    ----------
    im : 2D array
        full image. Use (xp, yp) to define where the star is.

    plate_scale : float
        [arcsec/pixel]

    mode : str
        can be "psf", "aperture", "sum", "core", "none"

    radius : float, int
        aperture radius

    xp, yp : float
        pixel coordinates of the centre of the star


    Returns
    -------
    flux : int, list

    """

    import scipy.optimize as opt

    n = radius
    if xp is None:
        xp = n
    if yp is None:
        yp = n

    #xp, yp = sky_to_pix(x, plate_scale), sky_to_pix(y, plate_scale)
    rr = np.arange(0, 2*n+1)
    xx, yy = np.meshgrid(rr, rr)

    sig = 3
    data = im[yp-n:yp+n+1, xp-n:xp+n+1]
    mx = np.max(data)
    initial_guess = (mx,n,n,sig,sig,0,0)

    try:
        popt, pcov = opt.curve_fit(twoD_Gaussian, (xx, yy), data.ravel(),
                                   p0=initial_guess)
        data_fitted = twoD_Gaussian((xx, yy), *popt).reshape((2*n+1,2*n+1))

    except:
        popt = np.zeros(7)
        data_fitted = np.zeros(data.shape)

    # xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    #print("amplitude, xo, yo, sigma_x, sigma_y, theta, offset\n", popt)

    height, floor = popt[0], popt[6]
    sig_x, sig_y = popt[3:5]
    vol = 2*np.pi*(height-floor)*sig_x*sig_y

    apertures = CircularAperture(popt[1:3], r=n)
    phot_table = aperture_photometry(data, apertures)

    #data.sum(), data_fitted.sum(), vol/0.57, phot_table["aperture_sum"][0]

    if mode == "psf":
        return data_fitted.sum()
    elif mode == "sum":
        return data.sum()
    elif mode == "core":
        return vol
    elif mode == "aperture":
        return phot_table["aperture_sum"][0]
    else:
        return data.sum(), data_fitted.sum(), vol, phot_table["aperture_sum"][0]


def subtract_psfs(image, psf, radius, x, y, e_limit=0.1, **kwargs):
        """
        Subtract PSFs from an image using linear regression

        Parameters
        ----------
        image : 2D array
            the image from which the PSFs with be subtracted

        psf :  2D array
            image of the PSF, doesn't need to be size-matched to the image

        radius : int
            radius of box around the PSF to be used for matching the height and
            base

        x, y : list, array
            pixel coordinates

        e_limit : float
            If the relative difference between thielslope fits is less than some
            value, reject the subtraction
            i.e. (m_high - m_low) / m < 2 * e_limit


        Returns
        -------
        im_new : 2D array

        results : list
            The results of the fit as returned by ``scipy.stats.linregress``
            i.e. gradient, intercept, r-value, p-value, err


        """

        im_new = np.copy(image)

        if np.shape(psf)[0] < 2 * np.shape(image)[0]:
            w, h = np.shape(image)
            pw, ph = np.shape(psf)
            pad_w = int(w-pw/2)+1
            psf_pad = np.pad(psf, pad_w, mode="constant")
        else:
            psf_pad = psf

        q = int(radius)
        cy, cx = np.where(psf_pad == psf_pad.max())
        cy, cx = cy[0], cx[0]

        psf_cutout = np.copy(psf_pad[cx-q : cx+q+1, cy-q : cy+q+1])
        psf_flat = psf_cutout.ravel()

        fit_results = []

        for xx, yy in zip(x, y):

            xii, yii = int(xx), int(yy)

            w, h = im_new.shape
            dx0, dx1 = xii, w-xii
            dy0, dy1 = yii, h-yii

            q1, q2, q3, q4 = min(q, dx0), min(q, dx1), min(q, dy0), min(q, dy1)
            im_cutout  = np.copy(im_new[xii-q1 : xii+q2+1, yii-q3 : yii+q4+1])
            im_flat = im_cutout.ravel()

            if len(psf_flat) != len(im_flat):
                fit_results += [[0]*7]
                continue


            m, c, r, p, e = linregress(psf_flat, im_flat)
            m, c, a, b = theilslopes(im_flat, psf_flat)
            e = 0.5*(b-a)/m

            # If it failes the null-hypothosis test - i.e. p > 0.1
            if p > 0.1:
                fit_results += [[0]*7]
                continue
            # If the relative difference between thielslope fits is less than
            #   some value, reject the subtraction
            # i.e. the (m_high - m_low) < 2 * m * e_limit
            if e > e_limit:
                fit_results += [[0]*7]
                continue
            # If the fitted slope is less then zero, forget the fit
            if m < 0:
                fit_results += [[0]*7]
                continue
            else:
                fit_results += [[m, c, r, p, e, a, b]]


            psf_cutout = np.copy(psf_pad[cx-dx0 : cx+dx1, cy-dy0 : cy+dy1])
            psf_cutout *= m

            im_new[xii-dx0 : xii+dx1, yii-dy0 : yii+dy1] -= psf_cutout

        return im_new, fit_results


def iter_psf_photometry(image, psf, radius, n_steps=5, **kwargs):
    """
    Iterates over an image, removing stars according to a scaled PSF

    Parameters
    ----------
    image : 2D array
        The image to investigate

    psf : 2D array
        Image of the offending PSF

    radius : int
        The radius of box used by ``subtract_psfs()``

    n_steps : int
        DAOStarFinder iterates from 1000*\sigma to 5*\sigma when looking for stars
        n_step is the number of steps in logspace between 1000 and 5 \sigma
        Default is 5


    Optional Parameters
    -------------------
    sigma_bins : array
        The sigma bins for setting

    verbose : bool

    Returns
    -------
    results : astropy.Table
        Table with DAOStarFinder results for all runs

    new_im : 2D array
        the resulting image after the stars found by DAOStarFinder have been
        subtracted

    """

    new_im = np.copy(image).astype(np.float32)

    tables = []

    if "sigma_bins" in kwargs.keys():
        sigma_bins = kwargs["sigma_bins"]
    else:
        sigma_bins = np.logspace(0.7,3, n_steps)[::-1]

    for nsig in sigma_bins:

        mean, median, std = sigma_clipped_stats(new_im, sigma=3.0, iters=5)
        std = np.sqrt(np.median(new_im))

        daofind = DAOStarFinder(fwhm=2.0, threshold=nsig*std)
        sources = daofind(new_im - median)

        if "verbose" in kwargs.keys():
            if kwargs["verbose"]:
                print("Found "+str(len(sources))+" sources above "+str(nsig)+"sigma")

        xs, ys, peaks = sources["ycentroid"], sources["xcentroid"], sources["peak"]
        ii = np.argsort(peaks)[::-1]

        # remove all the stars in this current SN bin
        new_im, fit_results = subtract_psfs(new_im, psf, radius, xs[ii], ys[ii],
                                            **kwargs)

        fit_tbl = Table(data=np.array(fit_results), names=["m", "c", "r", "p",
                                                           "merr", "mlow", "mhigh"])
        idx_tbl = Table(data=[ii], names=["ii"])
        sources = hstack([sources[ii], fit_tbl, idx_tbl])

        mask = fit_tbl["m"] > 0
        tables += [sources[mask]]

    results = vstack(tables)

    return results, new_im
