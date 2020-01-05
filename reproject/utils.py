from concurrent import futures
import numpy as np

import astropy.nddata
from astropy.io import fits
from astropy.io.fits import CompImageHDU, HDUList, Header, ImageHDU, PrimaryHDU
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS

__all__ = ['parse_input_data', 'parse_input_weights', 'parse_output_projection']


def parse_input_data(input_data, hdu_in=None):
    """
    Parse input data to return a Numpy array and WCS object.
    """

    if isinstance(input_data, str):
        return parse_input_data(fits.open(input_data), hdu_in=hdu_in)
    elif isinstance(input_data, HDUList):
        if hdu_in is None:
            if len(input_data) > 1:
                raise ValueError("More than one HDU is present, please specify "
                                 "HDU to use with ``hdu_in=`` option")
            else:
                hdu_in = 0
        return parse_input_data(input_data[hdu_in])
    elif isinstance(input_data, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_data.data, WCS(input_data.header)
    elif isinstance(input_data, tuple) and isinstance(input_data[0], np.ndarray):
        if isinstance(input_data[1], Header):
            return input_data[0], WCS(input_data[1])
        else:
            return input_data
    elif isinstance(input_data, astropy.nddata.NDDataBase):
        return input_data.data, input_data.wcs
    else:
        raise TypeError("input_data should either be an HDU object or a tuple "
                        "of (array, WCS) or (array, Header)")


def parse_input_weights(input_weights, hdu_weights=None):
    """
    Parse input weights to return a Numpy array.
    """

    if isinstance(input_weights, str):
        return parse_input_data(fits.open(input_weights), hdu_weights=hdu_weights)
    elif isinstance(input_weights, HDUList):
        if hdu_weights is None:
            if len(input_weights) > 1:
                raise ValueError("More than one HDU is present, please specify "
                                 "HDU to use with ``hdu_weights=`` option")
            else:
                hdu_weights = 0
        return parse_input_data(input_weights[hdu_weights])
    elif isinstance(input_weights, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_weights.data
    elif isinstance(input_weights, np.ndarray):
        return input_weights
    else:
        raise TypeError("input_weights should either be an HDU object or a Numpy array")


def parse_output_projection(output_projection, shape_out=None, output_array=None):

    if shape_out is None:
        if output_array is not None:
            shape_out = output_array.shape
    elif shape_out is not None and output_array is not None:
        if shape_out != output_array.shape:
            raise ValueError("shape_out does not match shape of output_array")

    if isinstance(output_projection, Header):
        wcs_out = WCS(output_projection)
        try:
            shape_out = [output_projection['NAXIS{}'.format(i + 1)]
                         for i in range(output_projection['NAXIS'])][::-1]
        except KeyError:
            if shape_out is None:
                raise ValueError("Need to specify shape since output header "
                                 "does not contain complete shape information")
    elif isinstance(output_projection, BaseHighLevelWCS):
        wcs_out = output_projection
        if shape_out is None:
            raise ValueError("Need to specify shape_out when specifying "
                             "output_projection as WCS object")
    elif isinstance(output_projection, str):
        hdu_list = fits.open(output_projection)
        shape_out = hdu_list[0].data.shape
        header = hdu_list[0].header
        wcs_out = WCS(header)
        hdu_list.close()
    else:
        raise TypeError('output_projection should either be a Header, a WCS '
                        'object, or a filename')

    if len(shape_out) == 0:
        raise ValueError("The shape of the output image should not be an "
                         "empty tuple")
    return wcs_out, shape_out


def block(reproject_func, input_data, wcs_out_sub, shape_out, return_footprint):
    array, footprint = reproject_func(input_data=input_data, output_projection=wcs_out_sub,
                                      shape_out=shape_out, return_footprint=return_footprint)
    return array, footprint

def reproject_blocked(reproject_func, block_size=(100,100), output_array=None, output_footprint=None,
                      parallel=0, **kwargs):
    print(kwargs)
    #array_in, wcs_in = parse_input_data(kwargs.get('input_data'), hdu_in=kwargs.get('hdu_in'))

    kwargs['wcs_out'], kwargs['shape_out'] = parse_output_projection(kwargs.get('output_projection'), shape_out=kwargs.get('shape_out'),
                                                 output_array=kwargs.get('output_array'))


    if kwargs.get('return_footprint') is None:
        kwargs['return_footprint'] = True


    if output_array is None:
        output_array = np.zeros(kwargs['shape_out'], dtype=float)
    if output_footprint is None and kwargs.get('return_footprint') != False:
        output_footprint = np.zeros(kwargs['shape_out'], dtype=float)

    proc_pool = futures.ProcessPoolExecutor()

    for imin in range(0, output_array.shape[0], block_size[0]):
        imax = min(imin + block_size[0], output_array.shape[0])
        print("reprojecting row " + str(imin))
        for jmin in range(0, output_array.shape[1], block_size[1]):
            print(jmin)
            jmax = min(jmin + block_size[1], output_array.shape[1])
            shape_out_sub = (imax - imin, jmax - jmin)
            wcs_out_sub = kwargs['wcs_out'].deepcopy()
            wcs_out_sub.wcs.crpix[0] -= jmin
            wcs_out_sub.wcs.crpix[1] -= imin

            #array_sub[:], footprint_sub = reprojct_func(input_data=kwargs['input_data'], output_projection=wcs_out_sub,
            #                shape_out=shape_out_sub,
            #                return_footprint=kwargs['return_footprint'])

            array_sub, footprint_sub = block(reproject_func=reproject_func, input_data=kwargs['input_data'], wcs_out_sub=wcs_out_sub,
                              shape_out=shape_out_sub, return_footprint=kwargs['return_footprint'])

            output_array[imin:imax, jmin:jmax] = array_sub[:]

            if kwargs['return_footprint']:
                output_footprint[imin:imax, jmin:jmax] = footprint_sub[:]

            # footprint[imin:imax, jmin:jmax] = footprint_sub

    if kwargs['return_footprint']:
        return output_array, output_footprint
    else:
        return output_array