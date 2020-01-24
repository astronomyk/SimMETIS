import numpy as np
from scipy.signal import convolve
from scipy.ndimage import zoom
from scipy.interpolate import griddata

from astropy.io import fits
from astropy.io import ascii as ioascii
from astropy.table import Table

from astropy import units as u

from . import utils
from . import image_plane_utils as imp_utils


class DataContainer(object):
    def __init__(self, filename=None, table=None, array_dict=None, **kwargs):

        if filename is None and "file_name" in kwargs:
            filename = kwargs["file_name"]

        filename = utils.find_file(filename)
        self.meta = {"filename" : filename,
                     "history" : []}
        self.meta.update(kwargs)

        self.headers = []
        self.table = None
        self._file = None

        if filename is not None:
            if self.is_fits:
                self._load_fits()
            else:
                self._load_ascii()

        if table is not None:
            self._from_table(table)

        if array_dict is not None:
            self._from_arrays(array_dict)

    def _from_table(self, table):
        self.table = table
        self.headers += [table.meta]
        self.meta.update(table.meta)
        self.meta["history"] += ["Table added directly"]

    def _from_arrays(self, array_dict):
        data = []
        colnames = []
        for key, val in array_dict.items():
            data += [val]
            colnames += [key]
        self.table = Table(names=colnames, data=data)
        self.headers += [None]
        self.meta["history"] += ["Table generated from arrays"]

    def _load_ascii(self):
        self.table = ioascii.read(self.meta["filename"])
        hdr_dict = utils.convert_table_comments_to_dict(self.table)
        if isinstance(hdr_dict, dict):
            self.headers += [hdr_dict]
        else:
            self.headers += [None]

        self.table.meta.update(hdr_dict)
        self.meta.update(hdr_dict)
        self.meta["history"] += ["ASCII table read from {}"
                                 "".format(self.meta["filename"])]

    def _load_fits(self):
        self._file = fits.open(self.meta["filename"])
        for ext in self._file:
            self.headers += [ext.header]

        self.meta.update(dict(self._file[0].header))
        self.meta["history"] += ["Opened handle to FITS file {}"
                                 "".format(self.meta["filename"])]

    def get_data(self, ext=0, layer=None):
        data_set = None
        if self.is_fits:
            if isinstance(self._file[ext], fits.BinTableHDU):
                data_set = Table.read(self._file[ext], format="fits")
            else:
                if self._file[ext].data is not None:
                    data_dims = len(self._file[ext].data.shape)
                    if data_dims == 3 and layer is not None:
                        data_set = self._file[ext].data[layer]
                    else:
                        data_set = self._file[ext].data
        else:
            data_set = self.table

        return data_set

    @property
    def is_fits(self):
        return utils.is_fits(self.meta["filename"])

    @property
    def data(self):
        data_set = None
        if self.is_fits:
            for ii in range(len(self._file)):
                data_set = self.get_data(ii)
                if data_set is not None:
                    break
        else:
            data_set = self.table

        return data_set


class FieldVaryingPSF(DataContainer):
    def __init__(self, **kwargs):
        super(FieldVaryingPSF, self).__init__(**kwargs)
        self.waveset, self.kernel_indexes = get_psf_wave_exts(self._file)
        self.valid_waverange = None
        self.current_ext = None
        self.current_data = None
        self.kernel = None
        self._strehl_imagehdu = None

        self.meta["SIM_FLUX_ACCURACY"] = 1E-2
        self.meta["SIM_SUB_PIXEL_FLAG"] = False
        self.meta['description'] = "Master psf cube from list"
        self.meta["Type"] = "FVPSF"
        self.meta["OBS_SCAO_NGS_OFFSET_X"] = 0
        self.meta["OBS_SCAO_NGS_OFFSET_Y"] = 0

        self.meta.update(kwargs)

    def apply_to(self, fov):
        if len(fov.fields) > 0:
            if fov.hdu.data is None:
                fov.view(self.meta["SIM_SUB_PIXEL_FLAG"])

            old_shape = fov.hdu.data.shape

            canvas = None
            kernels_masks = self.get_kernel(fov)
            for kernel, mask in kernels_masks:

                sum_kernel = np.sum(kernel)
                if abs(sum_kernel - 1) > self.meta["SIM_FLUX_ACCURACY"]:
                    kernel /= sum_kernel

                new_image = convolve(fov.hdu.data, kernel, mode="same")
                if canvas is None:
                    canvas = np.zeros(new_image.shape)

                if mask is not None:
                    new_mask =  convolve(mask, kernel, mode="same")
                    canvas += new_image * new_mask
                else:
                    canvas = new_image

            new_shape = canvas.shape
            fov.hdu.data = canvas

            # ..todo: careful with which dimensions mean what
            if "CRPIX1" in fov.hdu.header:
                fov.hdu.header["CRPIX1"] += (new_shape[0] - old_shape[0]) / 2
                fov.hdu.header["CRPIX2"] += (new_shape[1] - old_shape[1]) / 2

            if "CRPIX1D" in fov.hdu.header:
                fov.hdu.header["CRPIX1D"] += (new_shape[0] - old_shape[0]) / 2
                fov.hdu.header["CRPIX2D"] += (new_shape[1] - old_shape[1]) / 2

        return fov

    def get_kernel(self, fov):
        # 0. get file extension
        # 1. pull out strehl map for fov header
        # 2. get number of unique psfs
        # 3. pull out those psfs
        # 4. if more than one, make masks for the fov on the fov pixel scale
        # 5. make list of tuples with kernel and mask

        fov_wave = 0.5 * (fov.meta["wave_min"] + fov.meta["wave_max"])
        jj = utils.nearest(self.waveset, fov_wave)
        ii = self.kernel_indexes[jj]
        if ii != self.current_ext:
            self.current_ext = ii
            self.current_data = self._file[ii].data
        kernel_pixel_scale = self._file[ii].header["CDELT1"]
        fov_pixel_scale = fov.hdu.header["CDELT1"]

        strl_hdu = self.strehl_imagehdu
        strl_cutout = get_strehl_cutout(fov.hdu.header, strl_hdu)

        layer_ids = np.round(np.unique(strl_cutout.data)).astype(int)
        if len(layer_ids) > 1:
            kernels = [self.current_data[ii] for ii in layer_ids]
            masks = [strl_cutout.data.T == ii for ii in layer_ids]          # there's a .T in here that I don't like
            self.kernel = [[krnl, msk] for krnl, msk in zip(kernels, masks)]
        else:
            self.kernel = [[self.current_data[layer_ids[0]], None]]

        # .. todo: re-scale kernel and masks to pixel_scale of FOV
        # .. todo: can this be put somewhere else to save on iterations?
        pix_ratio = fov_pixel_scale / kernel_pixel_scale
        if abs(pix_ratio - 1) > self.meta["SIM_FLUX_ACCURACY"]:
            for ii in range(len(self.kernel)):
                self.kernel[ii][0] = resize_array(self.kernel[ii][0], pix_ratio)

        # import matplotlib.pyplot as plt
        # plt.imshow(strl_cutout.data.T)
        # plt.show()

        return self.kernel

    @property
    def strehl_imagehdu(self):
        return self.get_strehl_imagehdu()

    def get_strehl_imagehdu(self, recalculate=False):
        if self._strehl_imagehdu is None or recalculate is True:
            dx = self.meta["OBS_SCAO_NGS_OFFSET_X"]
            dy = self.meta["OBS_SCAO_NGS_OFFSET_Y"]

            ecat = self._file[0].header["ECAT"]
            if isinstance(self._file[ecat], fits.ImageHDU):
                self._strehl_imagehdu = self._file[ecat]

            elif isinstance(self._file[ecat], fits.BinTableHDU):
                strl_hdu = make_strehl_map_from_table(self._file[ecat],
                                                      offset=(dx, dy))
                self._strehl_imagehdu = strl_hdu

        return self._strehl_imagehdu

    @property
    def info(self):
        return self.meta


class PoorMansFOV:
    """

    Parameters
    ----------
    chip : simmetis.detector.Chip
    wave_min, wave_max : float
        [um]

    """
    def __init__(self, chip, wave_min, wave_max):
        self.hdu = fits.ImageHDU()
        self.hdu.header["CRVAL1"] = chip.x_cen / 3600.
        self.hdu.header["CRVAL2"] = chip.y_cen / 3600.
        self.hdu.header["CRPIX1"] = chip.naxis1 / 2.
        self.hdu.header["CRPIX2"] = chip.naxis2 / 2.
        self.hdu.header["CDELT1"] = chip.pix_res / 3600.
        self.hdu.header["CDELT2"] = chip.pix_res / 3600.
        self.hdu.header["CTYPE1"] = "RA---TAN"
        self.hdu.header["CTYPE2"] = "DEC---TAN"
        self.hdu.header["CUNIT1"] = "deg"
        self.hdu.header["CUNIT2"] = "deg"
        self.hdu.header["NAXIS"] = 2
        self.hdu.header["NAXIS1"] = chip.naxis1
        self.hdu.header["NAXIS2"] = chip.naxis2

        self.meta = {"wave_min": wave_min, "wave_max": wave_max}


def get_psf_wave_exts(hdu_list):
    """
    Returns a dict of {extension : wavelength}
    Parameters
    ----------
    hdu_list
    Returns
    -------
    wave_set, wave_ext
    """

    if not isinstance(hdu_list, fits.HDUList):
        raise ValueError("psf_effect must be a PSF object: {}"
                         "".format(type(hdu_list)))

    wave_ext = [ii for ii in range(len(hdu_list))
                if "WAVE0" in hdu_list[ii].header]
    wave_set = [hdu.header["WAVE0"] for hdu in hdu_list
                if "WAVE0" in hdu.header]
    # wave_set = utils.quantify(wave_set, u.um)

    return wave_set, wave_ext


def make_strehl_map_from_table(tbl, pixel_scale=1*u.arcsec, offset=(0, 0)):
    x = tbl.data["x"] + offset[0]
    y = tbl.data["y"] + offset[1]
    map = griddata(np.array([x, y]).T, tbl.data["layer"],
                   np.array(np.meshgrid(np.arange(-25, 26),
                                        np.arange(-25, 26))).T,
                   method="nearest")

    hdr = imp_utils.header_from_list_of_xy(np.array([-25, 25]) / 3600.,
                                           np.array([-25, 25]) / 3600.,
                                           pixel_scale=1/3600)

    map_hdu = fits.ImageHDU(header=hdr, data=map)

    return map_hdu


def get_strehl_cutout(fov_header, strehl_imagehdu):

    image = np.zeros((fov_header["NAXIS1"], fov_header["NAXIS2"]))
    canvas_hdu = fits.ImageHDU(header=fov_header, data=image)
    canvas_hdu = imp_utils.add_imagehdu_to_imagehdu(strehl_imagehdu,
                                                    canvas_hdu, order=0,
                                                    conserve_flux=False)
    canvas_hdu.data = canvas_hdu.data.astype(int)

    return canvas_hdu


def resize_array(image, scale_factor, order=1):
    sum_image = np.sum(image)
    image = zoom(image, scale_factor, order=order)
    image = np.nan_to_num(image, copy=False)        # numpy version >=1.13
    sum_new_image = np.sum(image)
    image *= sum_image / sum_new_image

    return image


def round_edges(kernel, edge_width=32, rounding_function="linear"):
    n = edge_width
    if "cos" in rounding_function:
        falloff = (np.cos(np.pi * np.arange(n) / (n-1)).reshape([1, n]) + 1) / 2
    elif "lin" in rounding_function:
        falloff = np.linspace(1, 0, n).reshape([1, n])
    elif "log" in rounding_function:
        falloff = np.logspace(0, -5, n).reshape([1, n])

    kernel[:n, :] *= falloff.T[::-1, :]
    kernel[-n:, :] *= falloff.T
    kernel[:, :n] *= falloff[:, ::-1]
    kernel[:, -n:] *= falloff

    return kernel
