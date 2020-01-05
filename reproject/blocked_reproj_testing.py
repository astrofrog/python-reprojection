from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from reproject import reproject_interp, reproject_exact, reproject_adaptive
import multiprocessing
import numpy as np
def main():



    hdu1 = fits.open(get_pkg_data_filename('galactic_center/gc_2mass_k.fits'))[0]
    hdu2 = fits.open(get_pkg_data_filename('galactic_center/gc_msx_e.fits'))[0]





    ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
    ax1.imshow(hdu1.data, origin='lower', vmin=-100., vmax=2000.)
    ax1.coords['ra'].set_axislabel('Right Ascension')
    ax1.coords['dec'].set_axislabel('Declination')
    ax1.set_title('2MASS K-band')

    ax2 = plt.subplot(1,2,2, projection=WCS(hdu2.header))
    ax2.imshow(hdu2.data, origin='lower', vmin=-2.e-4, vmax=5.e-4)
    ax2.coords['glon'].set_axislabel('Galactic Longitude')
    ax2.coords['glat'].set_axislabel('Galactic Latitude')
    ax2.coords['glat'].set_axislabel_position('r')
    ax2.coords['glat'].set_ticklabel_position('r')
    ax2.set_title('MSX band E')



    array, footprint = reproject_adaptive(hdu2, hdu1.header)

    ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
    ax1.imshow(array, origin='lower', vmin=-2.e-4, vmax=5.e-4)
    ax1.coords['ra'].set_axislabel('Right Ascension')
    ax1.coords['dec'].set_axislabel('Declination')
    ax1.set_title('Reprojected MSX band E image')

    ax2 = plt.subplot(1,2,2, projection=WCS(hdu1.header))
    ax2.imshow(footprint, origin='lower', vmin=0, vmax=1.5)
    ax2.coords['ra'].set_axislabel('Right Ascension')
    ax2.coords['dec'].set_axislabel('Declination')
    ax2.coords['dec'].set_axislabel_position('r')
    ax2.coords['dec'].set_ticklabel_position('r')
    ax2.set_title('MSX band E image footprint')

    #plt.show()


    from reproject.utils import reproject_blocked


    func_args ={}
    func_args['input_data'] = hdu2
    func_args['output_projection'] = hdu1.header
    block_result, blockprint = reproject_blocked(reproject_adaptive, input_data=hdu2, output_projection=hdu1.header, block_size=(100,100))


    ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
    ax1.imshow(block_result, origin='lower', vmin=-2.e-4, vmax=5.e-4)
    ax1.coords['ra'].set_axislabel('Right Ascension')
    ax1.coords['dec'].set_axislabel('Declination')
    ax1.set_title('Reprojected MSX band E image')


    ax2 = plt.subplot(1,2,2, projection=WCS(hdu1.header))
    ax2.imshow(blockprint, origin='lower', vmin=0, vmax=1.5)
    ax2.coords['ra'].set_axislabel('Right Ascension')
    ax2.coords['dec'].set_axislabel('Declination')
    ax2.coords['dec'].set_axislabel_position('r')
    ax2.coords['dec'].set_ticklabel_position('r')
    ax2.set_title('MSX band E image footprint')

    array_closenesss = np.isclose(array, block_result, equal_nan=True)
    if np.any(array_closenesss == False):
        print("arrays not close!")
        plt.figure()
        plt.imshow(array_closenesss)
        plt.show()

    print_closenesss = np.isclose(footprint, blockprint, equal_nan=True)
    if np.any(print_closenesss == False):
        print("footprints not close!")
        plt.figure()
        plt.imshow(print_closenesss)
        plt.show()



    plt.show()



if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()