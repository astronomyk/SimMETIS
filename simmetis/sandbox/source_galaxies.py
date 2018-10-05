"""
A bunch of helper functions to generate galaxies in SimMETIS
"""

import scipy.ndimage as spi
from .spectral import EmissionCurve
from .source import get_SED_names, SED, source_from_image

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
                   width=1024, height=1024, x_offset=0, y_offset=0, oversample=1):
    """
    Returns a 2D array with a normailised sersic profile

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
    Most units are in [pixel] in this function. This differs from :func:`.galaxy`
    where parameter units are in [arcsec] or [pc]

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
    thresh = np.max(np.array([img[0,:], img[-1,:], img[:,0], img[:,-1]]).flatten())
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

    params = {"n"           : n,
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
