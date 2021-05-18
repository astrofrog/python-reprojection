# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
from astropy.utils import deprecated_renamed_argument

from ..utils import parse_input_data, parse_output_projection, reproject_blocked
from .core import _reproject_full

__all__ = ['reproject_interp']

ORDER = {}
ORDER['nearest-neighbor'] = 0
ORDER['bilinear'] = 1
ORDER['biquadratic'] = 2
ORDER['bicubic'] = 3


@deprecated_renamed_argument('independent_celestial_slices', None, since='0.6')
def reproject_interp(input_data, output_projection, shape_out=None, hdu_in=0,
                     order='bilinear', independent_celestial_slices=False,
                     output_array=None, return_footprint=True, output_footprint=None, block_size=None, parallel=False):
    """
    Reproject data to a new projection using interpolation (this is typically
    the fastest way to reproject an image).

    Parameters
    ----------
    input_data
        The input data to reproject. This can be:

            * The name of a FITS file
            * An `~astropy.io.fits.HDUList` object
            * An image HDU object such as a `~astropy.io.fits.PrimaryHDU`,
              `~astropy.io.fits.ImageHDU`, or `~astropy.io.fits.CompImageHDU`
              instance
            * A tuple where the first element is a `~numpy.ndarray` and the
              second element is either a `~astropy.wcs.WCS` or a
              `~astropy.io.fits.Header` object
            * An `~astropy.nddata.NDData` object from which the ``.data`` and
              ``.wcs`` attributes will be used as the input data.

    output_projection : `~astropy.wcs.WCS` or `~astropy.io.fits.Header`
        The output projection, which can be either a `~astropy.wcs.WCS`
        or a `~astropy.io.fits.Header` instance.
    shape_out : tuple, optional
        If ``output_projection`` is a `~astropy.wcs.WCS` instance, the
        shape of the output data should be specified separately.
    hdu_in : int or str, optional
        If ``input_data`` is a FITS file or an `~astropy.io.fits.HDUList`
        instance, specifies the HDU to use.
    order : int or str, optional
        The order of the interpolation. This can be any of the
        following strings:

            * 'nearest-neighbor'
            * 'bilinear'
            * 'biquadratic'
            * 'bicubic'

        or an integer. A value of ``0`` indicates nearest neighbor
        interpolation.
    output_array : None or `~numpy.ndarray`
        An array in which to store the reprojected data.  This can be any numpy
        array including a memory map, which may be helpful when dealing with
        extremely large files.
    return_footprint : bool
        Whether to return the footprint in addition to the output array.
    output_footprint : None or `~numpy.ndarray`
        An array in which to store the reprojected footprint.  This can be any numpy
        array including a memory map, which may be helpful when dealing with
        extremely large files.
    block_size : None or tuple of (int, int)
        If not none, a blocked projection will be performed where the output space is
        reprojected to one block at a time, this is useful for memory limited scenarios
        such as dealing with very large arrays or high resolution output spaces.
    parallel : bool or int

    Returns
    -------
    array_new : `~numpy.ndarray`
        The reprojected array
    footprint : `~numpy.ndarray`
        Footprint of the input array in the output array. Values of 0 indicate
        no coverage or valid values in the input image, while values of 1
        indicate valid values.
    """

    array_in, wcs_in = parse_input_data(input_data, hdu_in=hdu_in)
    wcs_out, shape_out = parse_output_projection(output_projection, shape_out=shape_out,
                                                 output_array=output_array)

    if isinstance(order, str):
        order = ORDER[order]

    parallel = True
    # if either of these are not default, it means a blocked method must be used
    if block_size is not None or parallel is not False:
        # if parallel is set but block size isn't, we'll choose
        # block size so each thread gets one block each
        if parallel is not False and block_size is None:
            block_size = tuple(dim // os.cpu_count() for dim in shape_out)
        return reproject_blocked(_reproject_full, array_in=array_in, wcs_in=wcs_in, wcs_out=wcs_out,
                                 shape_out=shape_out, output_array=output_array, parallel=parallel,
                                 block_size=block_size, output_footprint=output_footprint)
    else:
        return _reproject_full(array_in, wcs_in, wcs_out, shape_out=shape_out, order=order,
                               array_out=output_array, return_footprint=return_footprint)
