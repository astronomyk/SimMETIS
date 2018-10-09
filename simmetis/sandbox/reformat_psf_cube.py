'''Reformat psf data cube to fits file with extensions

Date: 2018-10-09
Author: Oliver Czoske

Purpose: We have received PSF from Markus Feldt (MPIA) in a FITS format
         that is not directly readable with SimCADO/SimMETIS. This script
         reformats the data to make them compatible.
'''

import sys
from astropy.io import fits
from astropy.wcs import WCS

def create_hdu(psfimg, wavelen, pixscale):
    """Create a fits extension for a psf

    Parameters
    ----------
    psfimg : np.array
        psf image
    wavelen : float
        wavelength (in um) at which psf was computed
    pixscale : float
        pixel scale (in milliarcsec)
    """
    naxis2, naxis1 = psfimg.shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['LINEAR', 'LINEAR']
    wcs.wcs.cunit = ['mas', 'mas']
    wcs.wcs.crpix = [naxis2/2 + 1, naxis1/2 + 1]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.cdelt = [pixscale, pixscale]

    header = fits.Header()
    header['WAVELEN'] = (wavelen, 'microns')
    header['PIXSCALE'] = (pixscale, 'milliarcsec')
    header.extend(wcs.to_header())

    hdu = fits.ImageHDU(psfimg, header,
                        name="PSF_{:.2f}um".format(wavelen))
    return hdu


def main(infile, outfile):
    """Main function"""

    infd = fits.open(infile)
    header = infd[0].header
    psfcube = infd[0].data
    infd.close()

    ## Extract wavelengths and pixel scales
    wavelengths = []
    pixscale = []
    for key, val in zip(header.keys(), header.values()):
        if key == "WAVELENG":
            wavelengths.append(val)
        elif key == "PIXSIZE":
            pixscale.append(val)

    ## Initialise hdu list for output
    hdulist = fits.HDUList()

    ## Create primary DU
    pduhead = fits.Header()
    pduhead['FILETYPE'] = 'Point Spread Functions'
    pduhead['AUTHOR'] = 'Oliver Czoske'
    pduhead['DATE'] = '2018-10-09'
    pduhead['SOURCE'] = "Markus Feldt, MPIA"
    pduhead["ORIGDATE"] = '2018-??-??'
    pduhead.extend(header)   ## append original header

    hdulist.append(fits.PrimaryHDU(header=pduhead))

    ## Step through wavelength slices, create extension for each
    for ilam in range(len(wavelengths)):
        hdulist.append(create_hdu(psfcube[ilam, 0, :, :],
                                  wavelengths[ilam],
                                  pixscale[ilam]))


    ## Write out
    hdulist.writeto(outfile, overwrite=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("""Usage: reformat_psf_cube <infile> <outfile>""")
        sys.exit()

    main(sys.argv[1], sys.argv[2])
