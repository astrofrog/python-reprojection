from concurrent import futures
import numpy as np

import astropy.nddata
from astropy.io import fits
from astropy.io.fits import CompImageHDU, HDUList, Header, ImageHDU, PrimaryHDU
from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseHighLevelWCS

__all__ = ['parse_input_data', 'parse_input_shape', 'parse_input_weights',
           'parse_output_projection']


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


def parse_input_shape(input_shape, hdu_in=None):
    """
    Parse input shape information to return an array shape tuple and WCS object.
    """

    if isinstance(input_shape, str):
        return parse_input_shape(fits.open(input_shape), hdu_in=hdu_in)
    elif isinstance(input_shape, HDUList):
        if hdu_in is None:
            if len(input_shape) > 1:
                raise ValueError("More than one HDU is present, please specify "
                                 "HDU to use with ``hdu_in=`` option")
            else:
                hdu_in = 0
        return parse_input_shape(input_shape[hdu_in])
    elif isinstance(input_shape, (PrimaryHDU, ImageHDU, CompImageHDU)):
        return input_shape.shape, WCS(input_shape.header)
    elif isinstance(input_shape, tuple) and isinstance(input_shape[0], np.ndarray):
        if isinstance(input_shape[1], Header):
            return input_shape[0].shape, WCS(input_shape[1])
        else:
            return input_shape[0].shape, input_shape[1]
    elif isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
        if isinstance(input_shape[1], Header):
            return input_shape[0], WCS(input_shape[1])
        else:
            return input_shape
    elif isinstance(input_shape, astropy.nddata.NDDataBase):
        return input_shape.data.shape, input_shape.wcs
    else:
        raise TypeError("input_shape should either be an HDU object or a tuple "
                        "of (array-or-shape, WCS) or (array-or-shape, Header)")


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


def _block(reproject_func, array_in, wcs_in, wcs_out_sub, shape_out, i_range, j_range,
           return_footprint):
    # i and j range must be passed through for multiprocessing to know where to reinsert patches
    res = reproject_func(array_in, wcs_in, wcs_out_sub,
                         shape_out=shape_out, return_footprint=return_footprint)

    return {'i': i_range, 'j': j_range, 'block': res}


def reproject_blocked(reproject_func, array_in, wcs_in, shape_out, wcs_out, block_size,
                      output_array=None,
                      return_footprint=True, output_footprint=None, parallel=True):
    if output_array is None:
        output_array = np.zeros(shape_out, dtype=float)
    if output_footprint is None and return_footprint:
        output_footprint = np.zeros(shape_out, dtype=float)

    # setup variables needed for multiprocessing if required
    proc_pool = None
    blocks_futures = []

    if parallel or type(parallel) is int:
        if type(parallel) is int:
            proc_pool = futures.ProcessPoolExecutor(max_workers=parallel)
        else:
            proc_pool = futures.ProcessPoolExecutor()

    sequential_blocks_done = 0
    for imin in range(0, output_array.shape[0], block_size[0]):
        imax = min(imin + block_size[0], output_array.shape[0])
        for jmin in range(0, output_array.shape[1], block_size[1]):
            jmax = min(jmin + block_size[1], output_array.shape[1])
            shape_out_sub = (imax - imin, jmax - jmin)
            wcs_out_sub = wcs_out.deepcopy()
            wcs_out_sub.wcs.crpix[0] -= jmin
            wcs_out_sub.wcs.crpix[1] -= imin

            if proc_pool is None:
                # if sequential input data and reinsert block into main array immediately
                completed_block = _block(reproject_func=reproject_func, array_in=array_in,
                                         wcs_in=wcs_in,
                                         wcs_out_sub=wcs_out_sub, shape_out=shape_out_sub,
                                         return_footprint=return_footprint,
                                         j_range=(jmin, jmax), i_range=(imin, imax))

                output_array[imin:imax, jmin:jmax] = completed_block['block'][0][:]
                if return_footprint:
                    output_footprint[imin:imax, jmin:jmax] = completed_block['block'][1][:]

                sequential_blocks_done += 1
            else:
                # if parallel just submit all work items and move on to waiting for them to be done
                future = proc_pool.submit(_block, reproject_func=reproject_func, array_in=array_in,
                                          wcs_in=wcs_in, wcs_out_sub=wcs_out_sub,
                                          shape_out=shape_out_sub,
                                          return_footprint=return_footprint, j_range=(jmin, jmax),
                                          i_range=(imin, imax))
                blocks_futures.append(future)

    # If a parallel implementation is being used that means the
    # blocks have not been reassembled yet and must be done now
    if proc_pool is not None:
        completed_future_count = 0
        for completed_future in futures.as_completed(blocks_futures):
            completed_block = completed_future.result()
            i_range = completed_block['i']
            j_range = completed_block['j']
            output_array[i_range[0]:i_range[1], j_range[0]:j_range[1]] \
                = completed_block['block'][0][:]

            if return_footprint:
                footprint_block = completed_block['block'][1][:]
                output_footprint[i_range[0]:i_range[1], j_range[0]:j_range[1]] = footprint_block

            completed_future_count += 1
            idx = blocks_futures.index(completed_future)
            # ensure memory used by returned data is freed
            completed_future._result = None
            del blocks_futures[idx], completed_future
        proc_pool.shutdown()
        del blocks_futures

    if return_footprint:
        return output_array, output_footprint
    else:
        return output_array
