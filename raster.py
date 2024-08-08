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

import rasterio
import warnings
import numpy as np
from matplotlib import pyplot, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')
rcParams['xtick.labelsize'] = 6
rcParams['ytick.labelsize'] = 6
fontprops = fm.FontProperties(size=6)


def read(filename):
    """
    .
    """
    dem = ElevationModel()
    dem.read(filename)
    return dem


class ElevationModel(object):
    """
    Main class for handling rasters.
    """

    def __init__(self, path=None):
        self.array = None
        self.raster = None
        if path is not None:
            self.read(path)

    def read(self, path):
        """
        Import a raster from path using rasterio.
        :param path: path to input raster.
        :Return: None
        """
        self.raster = rasterio.open(path)

    def write(self, path):
        """

        :param path:
        """
        with rasterio.Env():
            profile = self.raster.profile
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(self.raster.read(1), 1)

    def display(self, colormap='jet', background=None, background_colormap="binary", scale=None):
        """
        Display the raster.
        :param colormap: the colormap of the raster.
        :param background: a raster object to be displayed as background (e.g. slopes, hillshade)
        :param background_colormap: the colormap of the background raster. Better to use greyscale colormaps.
        :param scale: size of scalebar in m. Should be used for rasters in UTM projection. Default is None (no scalebar).
        :Return: None
        """

        figure = pyplot.figure(facecolor='w')
        ax = figure.add_subplot(111)
        # if self.algorithm is not None:
        #     title = self.algorithm
        #     if self.kernel is not None:
        #         title = title + ' (ksize = ' + str(self.kernel) + ' pixels)'
        #     ax.set_title(title, fontsize=10)
        extent = [self.raster.bounds[0], self.raster.bounds[2], self.raster.bounds[1], self.raster.bounds[3]]
        mask = self.raster.read(1) == self.raster.nodata
        if background is not None:
            image = np.ma.masked_where(mask, background.array)
            ax.imshow(image, cmap=background_colormap, extent=extent)
            a = 0.5
        else:
            a = 1
        image = np.ma.masked_where(mask, self.raster.read(1))
        vmin, vmax = np.nanpercentile(self.raster.read(1), 2), np.nanpercentile(self.raster.read(1), 98)
        cmap = ax.imshow(image, cmap=colormap, alpha=a, vmin=vmin, vmax=vmax, extent=extent)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = figure.colorbar(cmap, cax=cax)

        if scale is not None:
            # l = int(self.pixelSize[0]*scale)
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
