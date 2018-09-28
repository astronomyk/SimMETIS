"""
A module that holds the functions needed to plot the increase in flux due to
non-zero blocking outside of a filter's wavelength range
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table

from .. import commands
from .. import optics
from .. import spectral
from .. import detector
from .. import source

__all__ = ["get_flux", "make_tophat_tcs", "make_wing_tbl", "get_region_names",
           "plot_flux_vs_wing", "get_filter_wings_flux", "get_total_wing_leakage",
           "required_transmission"]



def get_flux(srcs, filts, **kwargs):
    """
    Returns the total flux from a series of sources through a series of filters

    Parameters
    ----------
    srcs : list of simmetis.Source objects
    filts : list of simmetis.TransmissionCurve objects

    Optional Parameters
    -------------------
    kwargs : any keyword-value pair for a simmetis.UserCommands object

    Returns
    -------
    fluxes : list of floats
        A list of fluxes per filter for each source

    """
    cmd = commands.UserCommands()
    cmd.cmds.update(kwargs)

    fluxes = []

    for filt in filts:
        cmd["INST_FILTER_TC"] = filt

        opt = optics.OpticalTrain(cmd)

        flux_i = []
        for src in srcs:
            fpa = detector.Detector(cmd)

            src.apply_optical_train(opt, fpa)
            flux_i += [np.sum(fpa.chips[0].array)]

        fluxes += [flux_i]

    fluxes = np.array(fluxes).T.tolist()

    if len(fluxes) == 1:
        fluxes = fluxes[0]
    return fluxes


def make_tophat_tcs(wavelengths, edges=[0.3, 3.0], dlam=0.001, lam_res=0.001):
    """
    Create a series of simmetis.TransmissionCurve objects for a filter and the red/blue wings

    Parameters
    ----------
    wavelengths : list, array
        [um] List of spectral region borders, i.e. the edges of a filter (e.g. (1.4, 1.8) for H band)
    edges : list, optional
        [um] The first and last edge for the wavelength regions. Default is (0.3, 3.0)um
    dlam : float, optional
        [um] The wavelength range over which the curve should rise from 0 to 1
        (e.g 1.4um to 1.4001um) Default is 1nm
    lam_res : float, optional
        [um] The resolution of the final transmission curves. Default is 1nm


    Returns
    -------
    tcs : list ofsimmetis.TransmissionCurves
        FIlter curves for the filter, plus "filter" curves for the red and blue wings

    """
    l = np.array(wavelengths)
    lam  = np.array([edges[0]] + np.array((l, l+dlam)).T.flatten().tolist() + [edges[1]])
    tcs=[]
    for i in range(0, len(lam), 2):
        val = np.zeros(len(lam))
        val[i:i+2] = 1
        tc = spectral.TransmissionCurve(lam=lam, val=val, lam_res=lam_res)
        tcs += [tc]
        print(val)
    print(lam)
    return tcs


def make_wing_tbl(fluxes, trans=[1E-2, 1E-3, 1E-4, 1E-5], filter_index=1, wavelengths=None):
    """
    Make an astropy.Table with the relative flux increase per wing for a series of blocking coeffients

    Flux is a list of 3 values: the first and third are the fluxed through the blue and red wings, the second
    middle value is the flux through the filter wavelength range. All fluxes should be for a 100% transmission.
    The function iterates over the values in `trans`  (transmission coefficients) to make a table with the
    relative fluxes between filter and the (blue/both/red) wing for a certain blocking coefficient.

    Parameters
    ----------
    fluxes : list of floats
        list with the fluxes for each wavelength range (e.g. red, blue wings + filter)
    trans : list of floats, optional
        the blocking coefficients for the wings
    filter_index : int, optional
        which flux in ``fluxes`` is for the actual filter
    wavelengths : list, array, optional
        [um] List of spectral region borders, i.e. the edges of a filter (e.g. (1.4, 1.8) for H band)

    Returns
    -------
    wing_tbl : astropy.Table

    """

    ref_flux = fluxes[filter_index]
    rel_flux = [[t*(np.sum(fluxes)-ref_flux)/ref_flux for t in trans]]
    rel_flux += [[t*fluxes[i]/ref_flux for t in trans] \
                 for i in range(len(fluxes)) if i != filter_index]

    reg_names = get_region_names(wavelengths, filter_index)

    wing_tbl = Table(data=[trans]+rel_flux,
                     names=["Wing_Factor", "Total"]+reg_names)
    return wing_tbl


def get_region_names(wavelengths=None, filter_index=1):
    """
    Generate a list of column names for a series of wavelength regions

    Parameters
    ----------
    wavelengths : list, array
        [um] List of spectral region borders, i.e. the edges of a filter (e.g. (1.4, 1.8) for H band)
    filter_index : int
        which wavelength is the beginning of the actual filter. Default=1
    """

    if isinstance(wavelengths, (np.ndarray, list, tuple)):
        l = wavelengths
        reg_names = [str(l[i])[:5]+" - "+str(l[i+1])[:5] \
                                      for i in range(len(wavelengths)-1) if i != filter_index-1]
        reg_names = ["<"+str(l[0])[:5]] + reg_names + [">"+str(l[-1])[:5]]
    else:
        reg_names = ["Region "+str(i) for i in range(len(fluxes)) if i != filter_index]

    return reg_names


def plot_flux_vs_wing(tbl, loc=4):
    """
    Plots the relative flux increase due to different wing blocking coefficients.

    Plots to the current axis. Subplot should be set beforehand and show() called afterwards

    Parameters
    ----------
    tbl : astropy.Table
        The output from make_wing_tbl
    loc : int, optional
        location for the legend

    """

    clrs = "kbr" if len(tbl.colnames[1:])==3 else "kbgrcym"
    for col, clr in zip(tbl.colnames[1:], clrs):
        plt.plot(tbl["Wing_Factor"], tbl[col], clr, label=col)
        plt.loglog()
        plt.grid("on")
        plt.legend(loc=loc)
        plt.xlabel("Wing transmission")
        plt.ylabel("Fractional flux increase")


def get_filter_wings_flux(wavelengths=(1.49, 1.78), spec_types=["A0V", "M5V"],
                           plot_atmo_bg=True, filter_name="My Filter",
                           fluxes=None, return_fluxes=True, make_plot=True,
                           **kwargs):

    """
    Plot the effect of non-zero blocking in the filter wings

    Parameters
    ----------
    wavelengths : list, array
        [um] the cut-off wavelengths for the filter. Default is (1.49, 1.78), i.e. H-band
    spec_type : list of strings, optional
        The spectral types to be used for the comparison
    plot_atmo_bg : bool
        Default is True. Plot only the atmospheric emission, i.e. no star
    filter_name : str
        For the plot title
    fluxes : array, list
        Save time - re-use the output from get_flux()
    return_fluxes : bool
        Return the fluxes from get_flux() for later use
    make_plot : bool
        If False, make not plot. Handy in combination with ``return_fluxes=True``

    Optional parameters
    -------------------
    Any keyword-value pair from a simmetis config file
    "SCOPE_USE_MIRROR_BG"   : "no",
    "INST_USE_AO_MIRROR_BG" : "no",
    "FPA_USE_NOISE"         : "no",
    "ATMO_USE_ATMO_BG"      : "no",
    "filter_index"          : 1,
    "trans"                 : [1E-2, 1E-3, 1E-4, 1E-5],
    "edges"                 : [0.3,3.0],
    "lam_res"               : 0.001,
    "loc"                   : 4,
    "num_plots_wide"        : 3


    Notes
    -----
    The current filter fluxes are calculated with all sources of noise turned off for
    the stars. No shot noise is taken into accoung either.
    For the atmosphere, all background emission is turned on, but still no shot noise is used
    """

    params = {"SCOPE_USE_MIRROR_BG"   : "no",
              "INST_USE_AO_MIRROR_BG" : "no",
              "FPA_USE_NOISE"         : "no",
              "ATMO_USE_ATMO_BG"      : "no",
              "filter_index"          : 1,
              "trans"                 : [1E-2, 1E-3, 1E-4, 1E-5],
              "edges"                 : [0.3,3.0],
              "lam_res"               : 0.001,
              "loc"                   : 4,
              "num_plots_wide"        : 3}
    params.update(kwargs)

    filts = make_tophat_tcs(wavelengths, edges=params["edges"], lam_res=params["lam_res"])


    src_stars = [source.star(spec_type=spt, mag=20, filter_name="H") for spt in spec_types]

    if fluxes is None:
        stars = get_flux(src_stars, filts, **params)
    else:
        stars = fluxes

    m = params["num_plots_wide"]
    n = int(np.ceil((len(spec_types)+1*plot_atmo_bg)/float(m)))

    if make_plot:
        plt.figure(figsize=(5*m,4*n+1.5))
        plt.suptitle(filter_name, fontsize=20)


        for i, ttl in zip(range(len(spec_types)), spec_types):

            star_tbl = make_wing_tbl(stars[i],
                                     filter_index=params["filter_index"],
                                     trans=params["trans"],
                                     wavelengths=wavelengths)

            plt.subplot(n, m, i+1)
            plot_flux_vs_wing(star_tbl, loc=params["loc"])
            plt.title(ttl)
            plt.ylim(1E-5,1E-1)

    if plot_atmo_bg:

        params.update({"ATMO_USE_ATMO_BG" : "yes",
                       "SCOPE_USE_MIRROR_BG" : "yes",
                       "INST_USE_AO_MIRROR_BG" : "yes",
                       "FPA_USE_NOISE" : "no"})

        src_sky = [source.empty_sky()]
        sky = get_flux(src_sky, filts, **params)

        sky_tbl = make_wing_tbl(sky,
                                filter_index=params["filter_index"],
                                trans=params["trans"],
                                wavelengths=wavelengths)

        if make_plot:
            plt.subplot(n, m, len(spec_types)+1)
            plot_flux_vs_wing(sky_tbl, loc=params["loc"])
            plt.title("Thermal BG (Atmo+Mirrors)")
            plt.ylim(1E-5,1E-1)

    if return_fluxes:
        return stars+[sky]


def get_total_wing_leakage(fluxes, wavelengths, transmission_coeffs=1E-3,
                           edges=[0.3,3.0], filter_index=1,
                           row_names=None, make_plot=True):
    """
    Get the total increase in flux due to leakage in the wings

    Parameters
    ----------
    fluxes : list, array
        the output from ``get_filter_wings_flux()``
    wavelengths : list, array
        [um] The borders of different wavelength regions. Generally the edges of the filter,
        but can also include different regions of the wings. E.g. for H-band, wavelengths=(1.49, 1.78)
    transmission_coeffs : float, list
        [0 .. 1] A single transmission coefficient for all region, or individual transmission
        coefficients  for each region specified by the borders given in ``wavelengths``
    edges : list, array
        [um] edges of the wavelength vector. Default is (0.3, 3.0)um
    filter_index : int
        which region contains the actual filter. Default is the second region, i.e. fluxes[1]
    row_names : list of strings
        The names of objects used for the fluxes. E.g. ("A0V", "M5V", "Sky")


    """

    fi = filter_index

    if isinstance(transmission_coeffs, (float, int)):
        trans = np.array([transmission_coeffs]*len(wavelengths))
    else:
        trans = np.array(transmission_coeffs)

    frac_list = [np.array([f/flux[fi] for f in flux if f != flux[fi]])*trans for flux in fluxes]
    reg_names = get_region_names(wavelengths, filter_index)

    if row_names is None:
        row_names = ["Object "+str(i) for i in range(len(fluxes))]

    sum_frac = [np.sum(c) for c in frac_list]

    data = [row_names]+np.array(frac_list).T.tolist()+[sum_frac]
    names = ["Name"]+reg_names+["Total"]

    tbl = Table(data=data, names=names)


    if make_plot:
        for fracs, spt, clr in zip(frac_list, row_names+["Sky"], "bgr"):
            fracs = list(fracs)
            fracs = fracs[:fi] + [0] + fracs[fi:]
            l = np.array(wavelengths)
            lam  = np.array([edges[0]] + np.array((l, l+0.001)).T.flatten().tolist() + [edges[1]])
            val = np.array([fracs]+[fracs]).T.flatten()
            plt.plot(lam, val, label=spt, c=clr)

            sf = np.sum(fracs)
            plt.plot(edges, [sf, sf], clr+":" )


        trans2 = trans[:fi].tolist() + [1] + trans[fi:].tolist()
        for i in range(len(trans2)):
            plt.text(np.average(lam[2*i:2*i+2]), 0.3, trans2[i], horizontalalignment="center")

        plt.semilogy()
        plt.legend(loc=3)
        #plt.grid("on")
        plt.xlim(0.5,3); plt.ylim(1E-5, 1E0)
        plt.xlabel("Wavelength [um]"); plt.ylabel("Leaked flux fraction per window")
        plt.title("For transmission coefficients "+str(trans))

    return tbl


def required_transmission(fluxes, total_flux_increase=1E-2, filter_index=1,
                          wavelengths=None, edges=(0.3, 3.0),
                          row_names=None, make_plot=False):
    """
    Generate a filter transmission curve which limits the total flux increase

    The total flux increase budget is split equally between the different regions defined by
    the wavelength borders.

    Parameters
    ----------
    fluxes : list, array
        the output from ``get_filter_wings_flux()``
    total_flux_increase : float
        the total flux increase due to leakage in the wings of the filter. Default is 1% flux increase
    filter_index : int
        which region contains the actual filter. Default is the second region, i.e. fluxes[1]
    wavelengths : list, array
        [um] The borders of different wavelength regions. Generally the edges of the filter, but can also
        include different regions of the wings. E.g. for H-band, wavelengths=(1.49, 1.78)
    row_names : list of strings
        The names of objects used for the fluxes. E.g. ("A0V", "M5V", "Sky")
    edges : list, array
        [um] edges of the wavelength vector. Default is (0.3, 3.0)um

    """

    fi = filter_index
    trans = total_flux_increase / (len(fluxes[0])-1)

    coeff_list = [[trans / (f/flux[fi]) for f in flux if f != flux[fi]] for flux in fluxes]
    reg_names = get_region_names(wavelengths, filter_index)

    if row_names is None:
        row_names = ["Object "+str(i) for i in range(len(fluxes))]

    data = [row_names]+np.array(coeff_list).T.tolist()
    names = ["Name"]+reg_names
    tbl = Table(data=data, names=names)

    if make_plot:
        for coeffs, spt in zip(coeff_list, row_names):
            coeffs = coeffs[:fi] + [1] + coeffs[fi:]
            l = np.array(wavelengths)
            lam  = np.array([edges[0]] + np.array((l, l+0.001)).T.flatten().tolist() + [edges[1]])
            val = np.array([coeffs]+[coeffs]).T.flatten()
            plt.plot(lam, val, label=spt)

        plt.semilogy()
        plt.legend(loc=2)
        plt.grid("on")
        plt.xlim(0.7,2.7)
        plt.ylim(1E-5, 1E-1)
        plt.xlabel("Wavelength [um]"); plt.ylabel("Required transmission")
        plt.title("To limit total wing flux leak to "+str(total_flux_increase*100)[:5]+"%")

    return tbl
