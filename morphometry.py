# -*- coding: utf-8 -*-
"""
/***************************************************************************
 pyGEM
 A stand-alone application for morpho-tectonic analyses.
                              -------------------
        begin                : 2017-03-08
        git sha              : $Format:%H$
        copyright            : (C) 2017 by Andreani Louis
        email                : andreani.louis@googlemail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import sys
import copy
import rasterio
import multiprocessing
import numpy as np
import cv2
from .raster import ElevationModel
from .utils import clock
from skimage import transform
from scipy import ndimage, interpolate
from skimage.morphology import disk
from skimage.filters import median as skimage_median
from matplotlib import pyplot, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import warnings

warnings.filterwarnings('ignore')
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
fontprops = fm.FontProperties(size=6)


def aspect(input_dem):
    """
    Downslope direction of the maximum rate of change in elevation from each cell to its neighbors.
    :param input_dem:
    :return: New instance of raster object.
    """

    print(clock() + ' Computing Aspect...')
    raster = rasterio.open(input_dem)

    asp = _process_aspect([raster.read(1), [raster.nodata, raster.res, 3]])
    asp[raster.read(1) == raster.nodata] = -1

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=raster.width, height=raster.height,
                       dtype='int16', crs=raster.crs, transform=raster.transform,
                       nodata=-1, )
    src.write(asp.astype('int16'), 1)
    del memfile

    print(clock() + ' DONE')

    return src


def hillshade(input_dem, azimuth=315, elev_angle=45, z_factor=0.01):
    """
    ArcGIS implementation.
    http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-hillshade-works.htm
    :param input_dem:
    :param azimuth: angular direction of the sun, measured from north in clockwise degrees from 0 to 360.
    :param elev_angle: illumination source above the horizon. The units are in degrees, from 0 (on the horizon) to 90.
    :param z_factor: elevation factor (lower values to smoothen the hillshade).
    :return: New instance of raster object.
    """

    print(clock() + ' Computing Hillshade...')
    raster = rasterio.open(input_dem)
    shd = _process_hillshade([raster.read(1), [azimuth, elev_angle, z_factor]])
    shd[raster.read(1) == raster.nodata] = 255

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=raster.width, height=raster.height,
                       dtype='uint8', crs=raster.crs, transform=raster.transform,
                       nodata=255, )
    src.write(shd.astype('uint8'), 1)
    del memfile

    print(clock() + ' DONE')

    return src


def slope(input_dem, integer=False):
    """
    ArcGIS implementation.
    http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-slope-works.htm
    :param input_dem:
    :param integer: If true return slopes as int16 instead of float32.
    :return: New instance of raster object.
    """

    print(clock() + ' Computing Slope...')
    input_raster = rasterio.open(input_dem)
    raster = input_raster.read(1).astype("float32")
    if input_raster.nodata is not None:
        raster[raster == input_raster.nodata] = np.nan
    dx = cv2.Sobel(raster, cv2.CV_64F, 1, 0, 3) / (8 * input_raster.res[0])
    dy = cv2.Sobel(raster, cv2.CV_64F, 0, 1, 3) / (8 * input_raster.res[1])
    slp = np.arctan(np.sqrt(dx ** 2 + dy ** 2)) * 180 / np.pi
    slp[np.isnan(slp)] = 0
    slp[raster == input_raster.nodata] = -1

    if integer is True:
        slp = slp.astype('int16')
        dtype = 'int16'
    else:
        slp = slp.astype('float32')
        dtype = 'float32'

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=input_raster.width, height=input_raster.height,
                       dtype=dtype, crs=input_raster.crs, transform=input_raster.transform,
                       nodata=-1, )
    src.write(slp, 1)
    del memfile

    print(clock() + ' DONE')

    return src


def tpi(input_dem, ksize=3, multiprocess=False, integer=False):
    """
    Topographic Position Index (difference in elevation between a pixel and its surrounding neighbors).
    :param input_dem:
    :param ksize: moving widow size. Must be odd.
    :param multiprocess: For extremely large DEMs. Use parallel processing if True.
    :param integer: If True return output array as int16 instead of original raster's datatype (default).
    :return: new instance of raster object.
    """

    print(clock() + ' Computing Topographic Position Index...')
    raster = rasterio.open(input_dem)
    if multiprocess is True:
        tiles, block_size = _raster_to_tiles(raster.read(1), buffer=int(ksize / 2.) + 21,
                                             parameters=[raster.nodata, ksize])
        pool = multiprocessing.Pool()
        results = pool.map(_process_tpi, tiles)
        pool.close()
        pool.join()
        topo_ind = _tiles_to_raster(results, raster.read(1).shape, int(ksize / 2.) + 21, block_size)
    else:
        topo_ind = _process_tpi([raster.read(1), [raster.nodata, ksize]])

    topo_ind[raster.read(1) == raster.nodata] = -9999
    if integer is True:
        topo_ind = topo_ind.astype('int16')
        dtype = 'int16'
    else:
        topo_ind = topo_ind.astype('float32')
        dtype = 'float32'

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=raster.width, height=raster.height,
                       dtype=dtype, crs=raster.crs, transform=raster.transform,
                       nodata=-9999, )
    src.write(topo_ind, 1)
    del memfile

    print(clock() + ' DONE')

    return src


def relief(input_dem, ksize=3, multiprocess=False):
    """
    Difference in elevation within a moving window (local relief aka relief amplitude).
    :param input_dem:
    :param ksize: moving widow size. Must be odd.
    :param multiprocess: For extremely large DEMs. Use parallel processing if True.
    :return: New instance of raster object.
    """

    print(clock() + ' Computing Local Relief...')
    raster = rasterio.open(input_dem)
    if multiprocess is True:
        tiles, block_size = _raster_to_tiles(raster.read(1), buffer=int(ksize / 2.) + 21,
                                             parameters=[raster.nodata, ksize])
        pool = multiprocessing.Pool()
        results = pool.map(_process_local_relief, tiles)
        pool.close()
        pool.join()
        local_relief = _tiles_to_raster(results, raster.read(1).shape, int(ksize / 2.) + 21, block_size)
    else:
        local_relief = _process_local_relief([raster.read(1), [raster.nodata, ksize]])

    local_relief[raster.read(1) == raster.nodata] = -9999

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=raster.width, height=raster.height,
                       dtype=raster.dtypes[0], crs=raster.crs, transform=raster.transform,
                       nodata=-9999, )
    src.write(local_relief, 1)
    del memfile

    print(clock() + ' DONE')

    return src


def hypsometry(input_dem, ksize=3, multiprocess=False, integer=False):
    """
    Hypsometric integral (aka Elevation Relief Ratio) within a moving window.
    This index represents the ratio between (Hmean-Hmin) and (Hmax-Hmin)
    :param input_dem:
    :param ksize: moving widow size in pixels. Must be odd.
    :param multiprocess: For large DEMs. Use parallel processing if True.
    :param integer: If True return output array as int16 instead of float32 (default).
    :return: New instance of raster object.
    """

    print(clock() + ' Computing Hypsometric Integral...')
    raster = rasterio.open(input_dem)
    if multiprocess is True:
        tiles, block_size = _raster_to_tiles(raster.read(1), buffer=int(ksize / 2.) + 1,
                                             parameters=[raster.nodata, ksize])
        pool = multiprocessing.Pool()
        results = pool.map(_process_hypsometry, tiles)
        pool.close()
        pool.join()
        hypso = _tiles_to_raster(results, raster.read(1).shape, int(ksize / 2.) + 1, block_size)
    else:
        hypso = _process_hypsometry([raster.read(1), [raster.nodata, ksize]])

    hypso[raster.read(1) == raster.nodata] = 255
    if integer is True:
        hypso = (hypso * 100).astype('uint8')
        dtype = 'uint8'
    else:
        hypso = hypso.astype('float32')
        dtype = 'float32'

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=raster.width, height=raster.height,
                       dtype=dtype, crs=raster.crs, transform=raster.transform,
                       nodata=255, )
    src.write(hypso, 1)
    del memfile

    print(clock() + ' DONE')

    return src


def roughness(input_dem, ksize=3, multiprocess=False):
    """
    Ratio between a topographic surface and a flat surface with same geographic extend.
    Grohmann, C.H. (2004). Morphometric analysis in Geographic Information Systems: applications
    of free software GRASS and R, Comput. Geosci., 30, 1055–1067.
    :param input_dem:
    :param ksize: moving widow size. Must be odd.
    :param multiprocess: For extremely large DEMs. Use parallel processing if True.
    :return: New instance of raster object.
    """

    print(clock() + ' Computing Surface Roughness...')
    raster = rasterio.open(input_dem)
    if multiprocess is True:
        tiles, block_size = _raster_to_tiles(raster.read(1), buffer=int(ksize / 2.) + 1,
                                             parameters=[raster.res, raster.nodata, ksize])
        pool = multiprocessing.Pool()
        results = pool.map(_process_roughness, tiles)
        pool.close()
        pool.join()
        rough = _tiles_to_raster(results, raster.read(1).shape, int(ksize / 2.) + 1, block_size)
    else:
        rough = _process_roughness([raster.read(1), [raster.res, raster.nodata, ksize]])

    rough[raster.read(1) == raster.nodata] = 0

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=raster.width, height=raster.height,
                       dtype='float32', crs=raster.crs, transform=raster.transform,
                       nodata=0, )
    src.write(rough, 1)
    del memfile

    print(clock() + ' DONE')

    return src


def median(input_dem, ksize=3, multiprocess=False):
    """
    Median filter.
    :param input_dem:
    :param ksize: moving widow radius in pixels. Must be odd.
    :param multiprocess: For large DEMs. Use parallel processing if True.
    :return: New instance of raster object.
    """

    print(clock() + ' Computing Median filter...')
    raster = rasterio.open(input_dem)
    if multiprocess is True:
        tiles, block_size = _raster_to_tiles(raster.read(1), buffer=int(ksize / 2.) + 1,
                                             parameters=[raster.nodata, ksize])
        pool = multiprocessing.Pool()
        results = pool.map(_process_median, tiles)
        pool.close()
        pool.join()
        med = _tiles_to_raster(results, raster.read(1).shape, int(ksize / 2.) + 1, block_size)
    else:
        med = _process_median([raster.read(1), [raster.nodata, ksize]])

    med[raster.read(1) == raster.nodata] = raster.nodata

    memfile = rasterio.MemoryFile()
    src = memfile.open(driver='GTiff', count=1, width=raster.width, height=raster.height,
                       dtype=raster.dtypes[0], crs=raster.crs, transform=raster.transform,
                       nodata=raster.nodata, )
    src.write(med, 1)
    del memfile

    print(clock() + ' DONE')

    return src


def display(input_raster, colormap='jet', background=None, background_colormap="binary", scale=None):
    """
    Display the raster.
    :param input_raster: 
    :param colormap: the colormap of the raster.
    :param background: a raster object to be displayed as background (e.g. slopes, hillshade)
    :param background_colormap: the colormap of the background raster. Better to use greyscale colormaps.
    :param scale: size of scalebar in m. Should be used for rasters in UTM projection. Default is None (no scalebar).
    :Return: None
    """
    if type(input_raster) == rasterio.io.DatasetReader or type(input_raster) == rasterio.io.DatasetWriter:
        raster = input_raster
    else:
        raster = rasterio.open(input_raster)
    figure = pyplot.figure(facecolor='w')
    ax = figure.add_subplot(111)
    # if algorithm is not None:
    #     title = algorithm
    #     if kernel is not None:
    #         title = title + ' (ksize = ' + str(kernel) + ' pixels)'
    #     ax.set_title(title, fontsize=10)
    extent = [raster.bounds[0], raster.bounds[2], raster.bounds[1], raster.bounds[3]]
    mask = raster.read(1) == raster.nodata
    if background is not None:
        image = np.ma.masked_where(mask, background.array)
        ax.imshow(image, cmap=background_colormap, extent=extent)
        a = 0.5
    else:
        a = 1
    image = np.ma.masked_where(mask, raster.read(1))
    vmin, vmax = np.nanpercentile(raster.read(1), 2), np.nanpercentile(raster.read(1), 98)
    cmap = ax.imshow(image, cmap=colormap, alpha=a, vmin=vmin, vmax=vmax, extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = figure.colorbar(cmap, cax=cax)

    if scale is not None:
        # l = int(pixelSize[0]*scale)
        if scale < 1000:
            t = str(int(scale)) + ' m'
        else:
            t = str(int(scale / 1000)) + ' km'
        scalebar = AnchoredSizeBar(ax.transData,
                                   scale, t, 'lower left',
                                   pad=0.5,
                                   borderpad=0.5,
                                   color='black',
                                   frameon=True,
                                   label_top=True,
                                   size_vertical=4,
                                   fontproperties=fontprops)

        ax.add_artist(scalebar)

    for tick in ax.get_yticklabels():
        tick.set_rotation(90)
        tick.set_verticalalignment('center')

    cbar.ax.tick_params(labelsize=6)
    ax.set_aspect('equal')
    ax.autoscale(enable=True, axis=u'both', tight=True)
    figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    return figure, ax


def write(input_raster, path):
    """
    Write the raster.
    :param input_raster: 
    :param path: 
    """
    with rasterio.Env():
        profile = input_raster.profile
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(input_raster.read(1), 1)


def _process_hillshade(inputs):
    raster = inputs[0]
    azimuth, elev_angle, z_factor = inputs[1]

    zenith = (90 - elev_angle) * np.pi / 180.0
    # azimuth = (360.0 - azimuth + 90) * np.pi / 180.0
    azimuth = (azimuth + 90) * np.pi / 180.0
    # compute slope
    dx = cv2.Sobel(raster, cv2.CV_64F, 1, 0, 3)
    dy = cv2.Sobel(raster, cv2.CV_64F, 0, 1, 3)
    slp = np.arctan(z_factor * np.sqrt(dx ** 2 + dy ** 2))
    # compute aspect
    asp = np.arctan2(dy, dx)
    asp[asp < 0] = asp[asp < 0] + 2 * np.pi
    asp[np.logical_and(dx == 0, dy > 0)] = np.pi / 2
    asp[np.logical_and(dx == 0, dy < 0)] = 2 * np.pi - np.pi / 2

    shd = ((np.cos(zenith) * np.cos(slp)) + (np.sin(zenith) * np.sin(slp) * np.cos(azimuth - asp)))
    shd = 254 * (shd - np.nanmin(shd)) / (np.nanmax(shd) - np.nanmin(shd))  # keep 255 for nodata pixels

    return shd.astype('uint8')


def _process_aspect(inputs):
    raster = inputs[0]
    no_data, pixel_size, ksize = inputs[1]

    raster = raster.astype("float32")
    raster[raster == no_data] = np.nan
    raster = np.rot90(raster)

    dx = cv2.Sobel(raster, cv2.CV_64F, 1, 0, ksize) / (8 * pixel_size[0])
    dy = cv2.Sobel(raster, cv2.CV_64F, 0, 1, ksize) / (8 * pixel_size[1])

    asp = np.arctan2(dy, dx)
    asp = (asp * 180 / np.pi) * (-1)
    asp[asp < 0] += 360

    asp = np.rot90(asp, axes=(1, 0))
    asp = asp * (-1) + 360
    asp[np.isnan(asp)] = -1

    return asp


def _sum_integral(array, ksize):
    """
    Compute sum of pixels within a kernel using image integral.
    :param array: Input numpy array.
    :param ksize: Input kernel size in pixels (i.e. a square of size window).
                  Kernel MUST be odd.
    :returns: summed grid.
    """
    array = array.astype('float64')
    padarray = np.pad(array, pad_width=int(ksize / 2.) + 1, constant_values=0)
    padarray = padarray[0:-1, 0:-1]
    integral = transform.integral_image(padarray)

    h = int(array.shape[0])
    w = int(array.shape[1])
    int_a = integral[0:h, 0:w]
    int_b = integral[0:h, ksize::]
    int_c = integral[ksize::, 0:w]
    int_d = integral[ksize::, ksize::]
    area_sum = int_a + int_d - int_b - int_c

    return area_sum


def _process_tpi(inputs):
    array = inputs[0]
    no_data, ksize = inputs[1]
    array = array.astype("float32")
    if no_data is not None:
        array[array == no_data] = 0
    sum_data = _sum_integral(array, ksize)
    is_data = (array > 0).astype("uint8")
    sum_ones = _sum_integral(is_data, ksize)
    mean = sum_data / sum_ones
    del sum_data, sum_ones, is_data
    topo_index = array - mean
    return topo_index


def _process_hypsometry(inputs):
    raster = inputs[0]
    no_data, ksize = inputs[1]

    raster = raster.astype("float32")

    # make sure that no data areas are not used for h_max filter
    if no_data is not None:
        raster[raster == no_data] = np.nanmin(raster) - 1
    h_max = ndimage.maximum_filter(raster, size=ksize)

    # make sure that no data areas are not used for h_min filter
    if no_data is not None:
        raster[raster == np.nanmin(raster)] = np.nanmax(raster) + 1
    h_min = ndimage.minimum_filter(raster, size=ksize)

    if no_data is not None:
        raster[raster == np.nanmax(raster)] = 0
    # sum_h = convolve(raster, np.ones((ksize, ksize))) # ndimage.convolve requires too much memory :(
    sum_h = _sum_integral(raster, ksize)
    raster = (raster != no_data).astype("uint8")
    # sum_ones = convolve(raster, np.ones((ksize, ksize)))
    sum_ones = _sum_integral(raster, ksize)
    h_mean = sum_h / sum_ones

    del sum_h, sum_ones

    hypso = (h_mean - h_min).astype("float32") / (h_max - h_min).astype("float32")

    del h_min, h_max, h_mean

    return hypso


def _process_roughness(inputs):
    array = inputs[0]
    pixel_size, no_data, ksize = inputs[1]
    array = array.astype("float32")
    if no_data is not None:
        array[array == no_data] = np.nan
    dx = cv2.Sobel(array, cv2.CV_64F, 1, 0, 3) / (8 * pixel_size[0])
    dy = cv2.Sobel(array, cv2.CV_64F, 0, 1, 3) / (8 * pixel_size[1])
    slopes = np.arctan(np.sqrt(dx ** 2 + dy ** 2)) * 180 / np.pi
    slopes[np.isnan(slopes)] = 0

    res = (np.abs(pixel_size[0]) + np.abs(pixel_size[1])) / 2
    raster_surfaces = res * np.sqrt((res ** 2) + (np.tan(slopes / (180 / np.pi)) * res) ** 2)
    raster_surfaces[array == no_data] = 0
    flat_surfaces = np.ones(array.shape) * (res ** 2)
    flat_surfaces[array == no_data] = 0

    rough = _sum_integral(raster_surfaces, ksize) / _sum_integral(flat_surfaces, ksize).astype("float32")

    return rough


def _process_local_relief(inputs):
    """
    Difference in elevation within a moving window.
    :return:
    """
    raster = inputs[0]
    no_data, ksize = inputs[1]

    # raster = raster.astype("float32")

    # make sure that no data areas are not used for h_max filter
    if no_data is not None:
        raster[raster == no_data] = np.nanmin(raster) - 1
    h_max = ndimage.maximum_filter(raster, size=ksize)

    # make sure that no data areas are not used for h_min filter
    if no_data is not None:
        raster[raster == np.nanmin(raster)] = np.nanmax(raster)
    h_min = ndimage.minimum_filter(raster, size=ksize)
    relief = h_max - h_min

    return relief


def _process_median(inputs):
    """
    Local maxima within a moving window.
    :return:
    """
    raster = inputs[0]
    no_data, ksize = inputs[1]

    # make sure that no data areas are not used for max filter
    med = skimage_median(raster, disk(ksize))
    med[med == no_data] = no_data

    return med


def _raster_to_tiles(array, buffer, parameters):
    """
    Subset a raster into tiles for multiprocessing.
    :param array: input array.
    :param buffer: buffer around the tiles (should be greater than the moving window)
    :param parameters: list of parameters to be passed with each tile.
    :return:
    """

    block_size = 5
    for i in [5, 9, 13, 17]:
        min_y = np.nanmin(np.diff(np.ceil(np.linspace(0, array.shape[0], i))))
        min_x = np.nanmin(np.diff(np.ceil(np.linspace(0, array.shape[1], i))))
        if min_y > buffer and min_x > buffer:
            block_size = i

    grid_rows = np.ceil(np.linspace(0, array.shape[0], block_size))
    grid_cols = np.ceil(np.linspace(0, array.shape[1], block_size))
    top = [int(i) if i == 0 else int(i) - buffer for i in grid_rows[0:-1]]
    bottom = [int(i) if i == np.nanmax(grid_rows) else int(i) + buffer for i in grid_rows[1::]]
    left = [int(i) if i == 0 else int(i) - buffer for i in grid_cols[0:-1]]
    right = [int(i) if i == np.nanmax(grid_cols) else int(i) + buffer for i in grid_cols[1::]]

    tiles = []
    for i in range(len(left)):
        for j in range(len(top)):
            subset_array = array[top[j]:bottom[j], left[i]:right[i]]
            tiles += [[subset_array, parameters]]

    return tiles, block_size


def _tiles_to_raster(tiles, shape, buffer, block_size):
    grid_rows = np.ceil(np.linspace(0, shape[0], block_size))
    grid_cols = np.ceil(np.linspace(0, shape[1], block_size))

    top = [int(i) if i == 0 else int(i) - buffer for i in grid_rows[0:-1]]
    bottom = [int(i) if i == np.nanmax(grid_rows) else int(i) + buffer for i in grid_rows[1::]]
    left = [int(i) if i == 0 else int(i) - buffer for i in grid_cols[0:-1]]
    right = [int(i) if i == np.nanmax(grid_cols) else int(i) + buffer for i in grid_cols[1::]]

    top = [int(i) if i == 0 else int(i) + buffer for i in top]
    bottom = [int(i) if i == np.nanmax(grid_rows) else int(i) - buffer for i in bottom]
    left = [int(i) if i == 0 else int(i) + buffer for i in left]
    right = [int(i) if i == np.nanmax(grid_cols) else int(i) - buffer for i in right]

    newarray = np.zeros(shape)
    n = 0
    for i in range(len(left)):
        for j in range(len(top)):
            if j == 0:
                y0 = 0
            else:
                y0 = buffer
            if j == len(left) - 1:
                y1 = tiles[n].shape[0]
            else:
                y1 = -buffer
            if i == 0:
                x0 = 0
            else:
                x0 = buffer
            if i == len(left) - 1:
                x1 = tiles[n].shape[1]
            else:
                x1 = -buffer
            newarray[top[j]:bottom[j], left[i]:right[i]] = tiles[n][y0:y1, x0:x1]
            n += 1

    return newarray
