**********************************
Regular celestial images and cubes
**********************************

One of the most common types of data to reproject are celestial images or
n-dimensional data (such as spectral cubes) where two of the axes are
celestial. There are several existing algorithms that can be used to
reproject such data:

* **Interpolation** (such as nearest-neighbor, bilinear, biquadratic
  interpolation and so on). This is the fastest algorithm and is suited to
  common use cases, but it is important to note that it is not flux
  conserving, and will not return optimal results if the input and output
  pixel sizes are very different.

* **Drizzling**, which consists of determining the exact overlap fraction of
  pixels, and optionally allows pixels to be rescaled before reprojection.
  More details can be found `here <here>`__. This method is more accurate
  than interpolation but is only suitable for images where the field of view
  is small so that pixels are well approximated by rectangles in world
  coordinates. This is slower but more accurate than interpolation for small
  fields of view.

* Computing the **exact overlap** of pixels on the sky by treating them as
  **four-sided spherical polygons** on the sky and computing spherical polygon
  intersection. This is essentially an exact form of drizzling, and should be
  appropriate for any field of view. However, this comes at a significant
  performance cost. This is the algorithm used by the `Montage
  <http://montage.ipac.caltech.edu/index.html>`_ package, and we have
  implemented it here using the same core algorithm.

Currently, this package implements interpolation and spherical polygon
intersection.

The reprojection/resampling is always done assuming that the image is in
**surface brightness units**. For example, if you have an image with a constant
value of 1, reprojecting the image to an image with twice as high resolution
will result in an image where all pixels are all 1.

.. _interpolation:

Interpolation
=============

Reprojection using interpolation can be done using the high-level
:func:`~reproject.reproject_interpolation` function::

    >>> from reproject import reproject_interpolation

This function takes two main arguments. The first argument is the image to
reproject, together with WCS information about the image. This can be either an
Astropy HDU object (specifically :class:`~astropy.io.fits.PrimaryHDU`,
:class:`~astropy.io.fits.ImageHDU`, or :class:`~astropy.io.fits.CompImageHDU`), or a tuple with two elements: a Numpy
array and either a :class:`~astropy.wcs.WCS` or a
:class:`~astropy.io.fits.Header` instance.

The second argument is the WCS information for the output image, which should
be specified either as a :class:`~astropy.wcs.WCS` or a
:class:`~astropy.io.fits.Header` instance. If this is specified as a
:class:`~astropy.wcs.WCS` instance, the ``shape_out`` argument to
:func:`~reproject.reproject_interpolation` should also be specified, and be
given the shape of the output image using the Numpy ``(ny, nx)`` convention
(this is because :class:`~astropy.wcs.WCS`, unlike
:class:`~astropy.io.fits.Header`, does not contain information about image
size).

.. note:: The input data can be n-dimensional as long as it has two celestial
          axes. However, at this time, we assume that the celestial axes are
          independent from the other axes, so that the reprojection is the
          same for all celestial slices.

As an example, we start off by opening a FITS file using Astropy::

    >>> from astropy.io import fits
    >>> hdu = fits.open('http://data.astropy.org/galactic_center/gc_msx_e.fits')[0]    # doctest: +REMOTE_DATA
    Downloading http://data.astropy.org/galactic_center/gc_msx_e.fits [Done]

The image is currently using a Plate Carée projection::

    >>> hdu.header['CTYPE1']   # doctest: +REMOTE_DATA
    'GLON-CAR'

We can create a new header using a Gnomonic projection::

    >>> new_header = hdu.header.copy()   # doctest: +REMOTE_DATA
    >>> new_header['CTYPE1'] = 'GLON-TAN'   # doctest: +REMOTE_DATA
    >>> new_header['CTYPE2'] = 'GLAT-TAN'   # doctest: +REMOTE_DATA

And finally we can call the :func:`~reproject.reproject_interpolation` function to reproject
the image::

    >>> from reproject import reproject_interpolation
    >>> new_image, footprint = reproject_interpolation(hdu, new_header)   # doctest: +REMOTE_DATA

The :func:`~reproject.reproject_interpolation` function returns two arrays -
the first is the reprojected input image, and the second is a 'footprint'
array which shows the fraction of overlap of the input image on the output
image grid. This footprint is 0 for output pixels that fall outside the input
image, 1 for output pixels that fall inside the input image.

We can then easily write out the reprojected image to a new FITS file::

    >>> fits.writeto('reprojected_image.fits', new_image, new_header)   # doctest: +REMOTE_DATA

The order of the interpolation can be controlled by setting the ``order=``
argument to either an integer or a string giving the order of the
interpolation. Supported strings include:

* ``'nearest-neighbor'``: zeroth order interpolation
* ``'bilinear'``: fisst order interpolation
* ``'biquadratic'``: second order interpolation
* ``'bicubic'``: third order interpolation

Drizzling
=========

Support for the drizzle algorithm will be implemented in future versions.

Spherical Polygon Intersection
==============================

Exact reprojection using the spherical polygon intersection can be done using
the high-level :func:`~reproject.reproject_flux_conserving` function::

    >>> from reproject import reproject_flux_conserving

The two first arguments, the input data and the output projection, should be
specified as for the :func:`~reproject.reproject_interpolation` function
described in `Interpolation`_. In addition, an optional ``parallel=`` option
can be used to control whether to parallelize the reprojection, and if so how
many cores to use (see :func:`~reproject.reproject_flux_conserving` for more
details).