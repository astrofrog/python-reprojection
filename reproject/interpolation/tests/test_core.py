# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from distutils.version import LooseVersion
from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.tests.helper import pytest

from ...array_utils import map_coordinates
from ..core_celestial import _reproject_celestial
from ..core_full import _reproject_full

# TODO: add reference comparisons


def test_reproject_slices_2d():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_ga.hdr'))
    header_out = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/gc_eq.hdr'))

    array_in = np.ones((700, 690))

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    _reproject_celestial(array_in, wcs_in, wcs_out, (660, 680))


def test_reproject_slices_3d():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    array_in = np.ones((10, 200, 180))

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2]]

    _reproject_celestial(array_in, wcs_in, wcs_out, (10, 160, 170))


def test_map_coordinates_rectangular():

    # Regression test for a bug that was due to the resetting of the output
    # of map_coordinates to be in the wrong x/y direction

    image = np.ones((3, 10))
    coords = np.array([(0, 1, 2), (1, 5, 9)])

    result = map_coordinates(image, coords)

    np.testing.assert_allclose(result, 1)


def test_reproject_celestial_3d():
    """
    Test both full_reproject and slicewise reprojection. We use a case where the
    non-celestial slices are the same and therefore where both algorithms can
    work.
    """

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    array_in = np.ones((3, 200, 180))

    # TODO: here we can check that if we change the order of the dimensions in
    # the WCS, things still work properly

    wcs_in = WCS(header_in)
    wcs_out = wcs_in.deepcopy()
    wcs_out.wcs.ctype = ['GLON-SIN', 'GLAT-SIN', wcs_in.wcs.ctype[2]]
    wcs_out.wcs.crval = [158.0501, -21.530282, wcs_in.wcs.crval[2]]
    wcs_out.wcs.crpix = [50., 50., wcs_in.wcs.crpix[2] + 0.5]

    out_full, foot_full = _reproject_full(array_in, wcs_in, wcs_out, (3, 160, 170))

    out_celestial, foot_celestial = _reproject_celestial(array_in, wcs_in, wcs_out, (3, 160, 170))

    np.testing.assert_allclose(out_full, out_celestial)
    np.testing.assert_allclose(foot_full, foot_celestial)


def test_slice_reprojection():
    """
    Test case where only the slices change and the celestial projection doesn't
    """

    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS1'] = 5
    header_in['NAXIS2'] = 4
    header_in['NAXIS3'] = 3

    header_out = header_in.copy()
    header_out['NAXIS3'] = 2
    header_out['CRPIX3'] -= 0.5

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = _reproject_full(inp_cube, wcs_in, wcs_out, (2, 4, 5))

    # we expect to be projecting from
    # inp_cube = np.arange(3, dtype='float').repeat(4*5).reshape(3,4,5)
    # to
    # inp_cube_interp = (inp_cube[:-1]+inp_cube[1:])/2.
    # which is confirmed by
    # map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant')
    # np.testing.assert_allclose(inp_cube_interp, map_coordinates(inp_cube.astype('float'), new_coords, order=1, cval=np.nan, mode='constant'))

    assert out_cube.shape == (2, 4, 5)
    assert out_cube_valid.sum() == 40.

    # We only check that the *valid* pixels are equal
    # but it's still nice to check that the "valid" array works as a mask
    np.testing.assert_allclose(out_cube[out_cube_valid.astype('bool')],
                               ((inp_cube[:-1] + inp_cube[1:]) / 2.)[out_cube_valid.astype('bool')])

    # Actually, I fixed it, so now we can test all
    np.testing.assert_allclose(out_cube, ((inp_cube[:-1] + inp_cube[1:]) / 2.))


def test_4d_fails():

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS'] = 4

    header_out = header_in.copy()
    w_in = WCS(header_in)
    w_out = WCS(header_out)

    array_in = np.zeros((2, 3, 4, 5))

    with pytest.raises(ValueError) as ex:
        x_out, y_out, z_out = _reproject_full(array_in, w_in, w_out, [2, 4, 5, 6])
    assert str(ex.value) == "Length of shape_out should match number of dimensions in wcs_out"


def test_inequal_wcs_dims():
    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_out = header_in.copy()
    header_out['CTYPE3'] = 'VRAD'
    header_out['CUNIT3'] = 'm/s'
    header_in['CTYPE3'] = 'STOKES'
    header_in['CUNIT3'] = ''

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    with pytest.raises(ValueError) as ex:
        out_cube, out_cube_valid = _reproject_full(inp_cube, wcs_in, wcs_out, (2, 4, 5))
    assert str(ex.value) == "Output WCS has a spectral component but input WCS does not"


def test_different_wcs_types():

    inp_cube = np.arange(3, dtype='float').repeat(4 * 5).reshape(3, 4, 5)
    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_out = header_in.copy()
    header_out['CTYPE3'] = 'VRAD'
    header_out['CUNIT3'] = 'm/s'
    header_in['CTYPE3'] = 'VELO'
    header_in['CUNIT3'] = 'm/s'

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    with pytest.raises(ValueError) as ex:
        out_cube, out_cube_valid = _reproject_full(inp_cube, wcs_in, wcs_out, (2, 4, 5))
    assert str(ex.value) == ("The input (VELO) and output (VRAD) spectral "
                             "coordinate types are not equivalent.")

# TODO: add a test to check the units are the same.


def test_reproject_3d_celestial_correctness_ra2gal():

    inp_cube = np.arange(3, dtype='float').repeat(7 * 8).reshape(3, 7, 8)

    header_in = fits.Header.fromtextfile(get_pkg_data_filename('../../tests/data/cube.hdr'))

    header_in['NAXIS1'] = 8
    header_in['NAXIS2'] = 7
    header_in['NAXIS3'] = 3

    header_out = header_in.copy()
    header_out['CTYPE1'] = 'GLON-TAN'
    header_out['CTYPE2'] = 'GLAT-TAN'
    header_out['CRVAL1'] = 158.5644791
    header_out['CRVAL2'] = -21.59589875
    # make the cube a cutout approximately in the center of the other one, but smaller
    header_out['NAXIS1'] = 4
    header_out['CRPIX1'] = 2
    header_out['NAXIS2'] = 3
    header_out['CRPIX2'] = 1.5

    header_out['NAXIS3'] = 2
    header_out['CRPIX3'] -= 0.5

    wcs_in = WCS(header_in)
    wcs_out = WCS(header_out)

    out_cube, out_cube_valid = _reproject_full(inp_cube, wcs_in, wcs_out,
                                               (header_out['NAXIS3'],
                                                header_out['NAXIS2'],
                                                header_out['NAXIS1']))

    assert out_cube.shape == (2, 3, 4)
    assert out_cube_valid.sum() == out_cube.size

    # only compare the spectral axis
    np.testing.assert_allclose(out_cube[:, 0, 0], ((inp_cube[:-1] + inp_cube[1:]) / 2.)[:, 0, 0])
