import numpy as np

from .core import _reproject_full


def _reproject_chunking(array_in, wcs_in, wcs_out, shape_out, order=1, array_out=None,
                    return_footprint=True, blocks=(100, 100)):
    """
        For a 2D image, reproject in chunks
        """

    # Create output arrays
    #array = np.zeros(shape_out, dtype=float)
    #footprint = np.zeros(shape_out, dtype=float)

    for imin in range(0, array_out.shape[0], blocks[0]):
        imax = min(imin + blocks[0], array_out.shape[0])
        for jmin in range(0, array_out.shape[1], blocks[1]):
            jmax = min(jmin + blocks[1], array_out.shape[1])
            shape_out_sub = (imax - imin, jmax - jmin)
            array_sub = np.zeros(shape_out_sub)
            wcs_out_sub = wcs_out.deepcopy()
            wcs_out_sub.wcs.crpix[0] -= jmin
            wcs_out_sub.wcs.crpix[1] -= imin
            print("reprojecting chunk at [" + str(imin) +", " + str(jmin) +"]")
            _reproject_full(array_in, wcs_in, wcs_out_sub,
                                                       shape_out=shape_out_sub, order=order,
                                                       array_out=array_sub, return_footprint=return_footprint)

            array_out[imin:imax, jmin:jmax] = array_sub[:]
            #footprint[imin:imax, jmin:jmax] = footprint_sub

    return array_out