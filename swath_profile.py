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
import geopandas as gpd
import numpy as np
import h5py
from scipy.special import binom
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as pyplot
import warnings
warnings.filterwarnings('ignore')


class SwathProfile(object):
    """
    Main class for creating swath topographic profiles.
    """

    def __init__(self, shp=None, hdf=None):

        self.distance = None
        self.elev_mean = None
        self.elev_max = None
        self.elev_min = None
        self.centerline_y = None
        self.centerline_x = None
        self.contour_y = None
        self.contour_x = None
        self.crs = None
        self.name = None
        self.baseline_x = None
        self.baseline_y = None
        self.grid = None
        self.spacing = None
        self.width = None

        if shp is not None:
            self.fromShapefile(shp)
        if hdf is not None:
            self.read_hdf(hdf)

    def save(self, filename):
        """
        Save current attributes to hdf file
        :param filename: path to hdf file.
        :return:
        """
        database = h5py.File(filename, 'w')
        if self.baseline_x is not None:
            database.create_dataset('baseline_x', data=self.baseline_x)
        if self.baseline_y is not None:
            database.create_dataset('baseline_y', data=self.baseline_y)
        if self.grid is not None:
            database.create_dataset('grid', data=self.grid)
        if self.spacing is not None:
            database.create_dataset('spacing', data=self.spacing)
        if self.width is not None:
            database.create_dataset('width', data=self.width)
        if self.crs is not None:
            database.create_dataset('crs', data=self.crs)
        database.close()

    def read_hdf(self, filename):
        """
        Import attributes from hdf file
        :param filename: path to hdf file.
        :return:
        """
        database = h5py.File(filename, 'r')
        if 'baseline_x' in database.keys():
            self.baseline_x = database['baseline_x'].value
        if 'baseline_y' in database.keys():
            self.baseline_y = database['baseline_y'].value
        if 'grid' in database.keys():
            self.grid = database['grid'].value
        if 'spacing' in database.keys():
            self.spacing = database['spacing'].value
        if 'width' in database.keys():
            self.width = database['width'].value
        if 'crs' in database.keys():
            self.crs = database['crs'].value
        database.close()

    def fromShapefile(self, path):
        """
        Import a shapefile containing a baseline. For now only one line is considered.
        :param path: path to input shapefile.
        :Return:
        """
        shp = gpd.read_file(path)
        points = np.vstack(shp.iloc[0].geometry.coords.xy).T
        self.baseline_x, self.baseline_y = points[:, 0], points[:, 1]
        self.crs = shp.crs

    def add_point_to_baseline(self, point):
        """
        Add point to current baseline.
        :param point: a tuple or list with x and y coordinates.
        :return:
        """
        self.baseline_x = np.hstack((self.baseline_x, point[0]))
        self.baseline_y = np.hstack((self.baseline_y, point[1]))

    def sample_grid(self, width, spacing, method='bezier', degree_conversion=False):
        """
        Create a grid of x,y,z coordinates as a base for sampling raster data.
        Note that z values must be extracted after creating the grid using 'extract' function.
        :param width: width of the swath in meters.
        :param spacing: spacing between the points in meters.
        :param method: 'bezier' (recommended) or 'chaikins' (under testing)
        :param degree_conversion: If true convert width and spacing to degrees (for geographic projections only).
        :return: a 3D array.
        """

        if isinstance(self.baseline_x, list):
            self.baseline_x = np.array(self.baseline_x)
            self.baseline_y = np.array(self.baseline_x)

        if degree_conversion is True and self.crs.is_geographic is True:
            dist_d = np.sqrt(np.diff(self.baseline_x[0:2]) ** 2 + np.diff(self.baseline_y[0:2]) ** 2)
            dist_m = _deg2m(self.baseline_x[0:2], self.baseline_y[0:2])
            factor = np.nanmean(dist_d / dist_m)
            width *= factor
            spacing *= factor

        self.width = width
        self.spacing = spacing

        if len(self.baseline_x) > 2:
            if method == 'bezier':
                x, y = _bezier_curve(self.baseline_x, self.baseline_y)
            else:
                x, y = zip(*_chaikins_corner_cutting(coords=[c for c in zip(self.baseline_x, self.baseline_y)],
                                                     refinements=10))

        else:
            x, y = self.baseline_x, self.baseline_y

        x, y = np.array(x), np.array(y)
        dist = np.sum(np.hypot(x[1::] - x[0:-1], y[1::] - y[0:-1]))
        n = int(dist / spacing)
        xi, yi = _curve_interpolation(x, y, n)

        grid_x, grid_y = _make_swath(xi, yi, self.width, self.spacing)

        self.grid = np.zeros((grid_x.shape[0], grid_x.shape[1], 3))
        self.grid[..., 0] = grid_x
        self.grid[..., 1] = grid_y

        self.get_contours()
        self.get_centerline()

    def get_contours(self):
        """
        Get the envelope of the sampled points from the swath profile.
        """
        self.contour_x = np.hstack((self.grid[0, :, 0], self.grid[-1, :, 0][::-1], self.grid[0, 0, 0]))
        self.contour_y = np.hstack((self.grid[0, :, 1], self.grid[-1, :, 1][::-1], self.grid[0, 0, 1]))

    def get_centerline(self):
        """
        Get the centerline of the swath profile.
        """
        ctr_id = int(self.grid.shape[0] / 2.)
        self.centerline_x, self.centerline_y = self.grid[ctr_id, :, 0], self.grid[ctr_id, :, 1]

    def get_raster_data(self, path):
        """
        Sample pixel values from a raster.
        :param path:
        """

        raster = rasterio.open(path)

        x, y = _geographic_to_cartesian(self.grid[..., 0], self.grid[..., 1], raster.transform)

        if np.nanmax(x) > raster.width or np.nanmax(y) > raster.height:
            if np.nanmax(x) > raster.width:
                xmax = int(np.ceil(np.nanmax(x))) + 1
            else:
                xmax = raster.width

            if np.nanmax(y) > raster.height:
                ymax = int(np.ceil(np.nanmax(y))) + 1
            else:
                ymax = raster.height
            array = np.zeros((ymax, xmax)) * np.nan
            array[0:raster.height, 0:raster.width] = raster.read(1)
            self.grid[..., 2] = _bilinear(x, y, array)
        else:
            self.grid[..., 2] = _bilinear(x, y, raster.read(1))

        cum_dist = np.cumsum(np.sqrt(np.diff(self.centerline_x) ** 2 + (np.diff(self.centerline_y)) ** 2))
        self.distance = np.hstack((0, cum_dist))
        self.elev_min = np.nanmin(self.grid[..., 2], axis=0)
        self.elev_max = np.nanmax(self.grid[..., 2], axis=0)
        self.elev_mean = np.nanmean(self.grid[..., 2], axis=0)

    def export_swath(self, path):
        """
        Export the enveloppe of the swath profile as polygon shapefile.
        :param path:
        """
        poly = [Polygon([item for item in zip(self.contour_x, self.contour_y)])]
        gdf = gpd.GeoDataFrame(geometry=poly, crs=self.crs)
        gdf.to_file(path)

    def export_centerline(self, path):
        """
        Export the centerline of the swath profile as line shapefile.
        :param path:
        """
        line = [LineString([item for item in zip(self.centerline_x, self.centerline_y)])]
        gdf = gpd.GeoDataFrame(geometry=line, crs=self.crs)
        gdf.to_file(path)


class _PointData(object):
    """
    Empty class to store some temporary point data inside.
    """

    def __init__(self, N):
        class _Coordinates(object):
            def __init__(self, x=[], y=[]):
                self.x = x
                self.y = y

        for i in range(N):
            setattr(self, str(i), _Coordinates())


def _bernstein(n, k):
    """Bernstein polynomial."""
    coeff = binom(n, k)

    def bpoly(x):
        return coeff * x ** k * (1 - x) ** (n - k)

    return bpoly


def _bezier(x, y, n=100):
    """connect two straight segments using bezier curve"""
    n_points = len(x)
    t = np.linspace(0, 1, n)
    curve = np.zeros((n, 2))
    points = list(zip(x, y))
    for i in range(n_points):
        curve += np.outer(_bernstein(n_points - 1, i)(t), points[i])
    x, y = curve.T
    return x, y


def _bezier_curve(x, y):
    """
    Convert a line made of straights segments to a curved one using bezier curves.
    :param x:
    :param y:
    :return:
    """

    # STEP 1: define points inbetween which the line corners will be curved.

    splits = _PointData(N=len(x) - 1)

    for pid in range(len(x) - 2):

        l1 = np.hypot(x[pid + 1] - x[pid], y[pid + 1] - y[pid])
        l2 = np.hypot(x[pid + 2] - x[pid + 1], y[pid + 2] - y[pid + 1])

        if l1 > l2:
            r = 1 - (l2 * 0.5 / l1)

            ind = [pid, pid + 1]
            s1_x = [x[ind][0], x[ind][0] + r * (x[ind][-1] - x[ind][0])]
            s1_y = [y[ind][0], y[ind][0] + r * (y[ind][-1] - y[ind][0])]
            ind = [pid + 1, pid + 2]
            s2_x = [np.nanmean(x[ind]), x[ind][-1]]
            s2_y = [np.nanmean(y[ind]), y[ind][-1]]

        else:
            r = 1 - (l1 * 0.5 / l2)

            ind = [pid, pid + 1]
            s1_x = [x[ind][0], np.nanmean(x[ind])]
            s1_y = [y[ind][0], np.nanmean(y[ind])]
            ind = [pid + 1, pid + 2]
            s2_x = [x[ind][0] + (1 - r) * (x[ind][-1] - x[ind][0]), x[ind][-1]]
            s2_y = [y[ind][0] + (1 - r) * (y[ind][-1] - y[ind][0]), y[ind][-1]]

        if len(splits.__dict__[str(pid)].x) == 0 or (s1_x[-1], s1_y[-1]) not in zip(
                splits.__dict__[str(pid)].x, splits.__dict__[str(pid)].y):
            splits.__dict__[str(pid)].x = list(splits.__dict__[str(pid)].x) + [s1_x[-1]]
            splits.__dict__[str(pid)].y = list(splits.__dict__[str(pid)].y) + [s1_y[-1]]
        if len(splits.__dict__[str(pid + 1)].x) == 0 or (s2_x[0], s2_y[0]) not in zip(
                splits.__dict__[str(pid + 1)].x, splits.__dict__[str(pid + 1)].y):
            splits.__dict__[str(pid + 1)].x = list(splits.__dict__[str(pid + 1)].x) + [s2_x[0]]
            splits.__dict__[str(pid + 1)].y = list(splits.__dict__[str(pid + 1)].y) + [s2_y[0]]

    # STEP 2: make a bezier curve for each triangle made of split points and initial corners.

    new_x = []
    new_y = []

    for pid in range(len(x) - 2):

        s1_x = [x[pid], splits.__dict__[str(pid)].x[-1]]
        s1_y = [y[pid], splits.__dict__[str(pid)].y[-1]]
        s2_x = [splits.__dict__[str(pid + 1)].x[0], x[pid + 2]]
        s2_y = [splits.__dict__[str(pid + 1)].y[0], y[pid + 2]]

        segment1 = [i for i in zip(s1_x, s1_y)]
        segment2 = [i for i in zip(s2_x, s2_y)]
        xi, yi = _intersect(segment1, segment2)

        xb, yb = _bezier([s1_x[1], xi, s2_x[0]], [s1_y[1], yi, s2_y[0]], n=1000)

        if pid == 0:
            new_x += s1_x
            new_y += s1_y

        new_x += list(xb[1::])
        new_y += list(yb[1::])

        if pid == len(x) - 3:
            new_x += s2_x[1::]
            new_y += s2_y[1::]

    return new_x, new_y


def _bilinear(x, y, grid):
    """
    Bilinear inerpolation of points located within a grid of values.
    :param x: x coordinates of the points.
    :param y: y coordinates of the points.
    :param grid: array containing the values to be sampled.
    :return: interpolated values.
    """

    xmin = (np.floor(x)).astype('int')
    xmax = xmin + 1
    ymin = (np.floor(y)).astype('int')
    ymax = ymin + 1
    int_up = ((xmax - x) / (xmax - xmin)) * grid[ymin, xmin] + ((x - xmin) / (xmax - xmin)) * grid[ymin, xmax]
    int_down = ((xmax - x) / (xmax - xmin)) * grid[ymax, xmin] + ((x - xmin) / (xmax - xmin)) * grid[ymax, xmax]
    z = ((ymax - y) / (ymax - ymin)) * int_up + ((y - ymin) / (ymax - ymin)) * int_down

    return z


def _chaikins_corner_cutting(coords, refinements=5):
    coords = np.array(coords)

    for _ in range(refinements):
        l = coords.repeat(2, axis=0)
        r = np.empty_like(l)
        r[0] = l[0]
        r[2::2] = l[1:-1:2]
        r[1:-1:2] = l[2::2]
        r[-1] = l[-1]
        coords = l * 0.75 + r * 0.25

    return coords


def _curve_interpolation(x, y, n_points):
    """
    Creates a set of equally spaced interpolated points along a 2D curve.
    Python adaptation of a Matlab function from John D'Errico:
    https://de.mathworks.com/matlabcentral/fileexchange/34874-interparc
    :param x: x coordinates of line (should be an array).
    :param y: y coordinates of line (should be an array).
    :param n_points: number of points to be interpolated.
    :return: interpolated x and y coordinates.
    """
    n = x.size  # number of points on the curve
    pxy = np.array((x, y)).T

    # Compute the chordal arclength of each segment.
    chordlen = np.sqrt(np.sum(np.diff(pxy, axis=0) ** 2, axis=1))
    # Normalize the arclengths to a unit total
    chordlen /= np.sum(chordlen)
    # cumulative arclength
    cumarc = np.append(0, np.cumsum(chordlen))

    n_points = np.transpose(np.linspace(0, 1, n_points))  # equally spaced in arclength
    tbins = np.digitize(n_points, cumarc)  # bin index in which each point of the curve is in
    # catch any problems at the ends
    tbins[np.where(np.logical_and(tbins <= 0, n_points <= 0))] = 1
    tbins[np.where(np.logical_and(tbins >= n, n_points >= 1))] = n - 1

    # interpolate
    s = np.divide((n_points - cumarc[tbins]), chordlen[tbins - 1])
    pt = pxy[tbins, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

    return pt[:, 0], pt[:, 1]


def _deg2m(x, y):
    """
    Get distance in meters between points with geographic coordinates (decimal degrees) using haversine formula.
    :param x:
    :param y:
    :return:
    """
    d = []
    for i in range(len(x) - 1):
        # Coordinates in decimal degrees (e.g. 2.89078, 12.79797)
        lon1, lat1 = x[i], y[i]
        lon2, lat2 = x[i + 1], y[i + 1]

        r = 6371000  # radius of Earth in meters
        phi_1 = np.radians(lat1)
        phi_2 = np.radians(lat2)

        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)

        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2.0) ** 2

        # c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        c = 2 * np.arcsin(np.sqrt(a))
        d += [r * c]  # output distance in meters
    return d


def _geographic_to_cartesian(x, y, geotransform):
    """
    Convertion of (x,y) points coordinates to cartesian coordinates.
    Geotransform from image in rasterio format.
    """
    x = (x - geotransform[2] - (geotransform[0] / 2)) / geotransform[0]
    y = (y - geotransform[5] - (geotransform[4] / 2)) / geotransform[4]

    return x, y


def _intersect(line1, line2):
    """
    Find the intersection point between two lines
    :param line1: coordinates of the first line ((x1,y1),(x2,y2))
    :param line2: coordinates of the second line ((x3,y3),(x4,y4))
    :return:
    """

    a = line1[0][0] * line1[1][1] - line1[1][0] * line1[0][1]
    b = line2[0][0] * line2[1][1] - line2[1][0] * line2[0][1]
    c = (line1[1][1] - line1[0][1]) * (line2[1][0] - line2[0][0]) - (line2[1][1] - line2[0][1]) * (
            line1[1][0] - line1[0][0])
    x = (a * (line2[1][0] - line2[0][0]) - b * (line1[1][0] - line1[0][0])) / c
    y = (a * (line2[1][1] - line2[0][1]) - b * (line1[1][1] - line1[0][1])) / c

    return x, y


def _make_swath(x, y, width, spacing):
    """
    Create a grid of values equally spaced from a line.
    :param x:
    :param y:
    :param width:
    :param spacing:
    :return:
    """
    half_width = int((width / 2) / spacing) * spacing
    nrows = int((width / 2) / spacing) * 2 + 1

    ind1 = range(0, len(x) - 1)
    ind2 = range(1, len(x))
    angle = np.arctan(
        (y[ind1] - y[ind2]).astype('float32') / (x[ind1] - x[ind2]).astype('float32'))
    angle[x[ind1] == x[ind2]] = 0.000001
    distance = np.zeros(angle.shape)
    distance[:] = half_width
    distance[x[ind1] > x[ind2]] = -half_width

    x1 = np.mean(np.vstack((x[ind1] + distance * np.sin(angle), x[ind2] + distance * np.sin(angle))), axis=0)
    x2 = np.mean(np.vstack((x[ind1] - distance * np.sin(angle), x[ind2] - distance * np.sin(angle))), axis=0)
    y1 = np.mean(np.vstack((y[ind1] - distance * np.cos(angle), y[ind2] - distance * np.cos(angle))), axis=0)
    y2 = np.mean(np.vstack((y[ind1] + distance * np.cos(angle), y[ind2] + distance * np.cos(angle))), axis=0)

    grid_x = np.vstack((x1, x2))
    x = np.arange(0., grid_x.shape[0])
    fit = interp1d(x, grid_x, axis=0)
    grid_x = fit(np.linspace(0, grid_x.shape[0] - 1, nrows))

    grid_y = np.vstack((y1, y2))
    x = np.arange(0., grid_y.shape[0])
    fit = interp1d(x, grid_y, axis=0)
    grid_y = fit(np.linspace(0, grid_y.shape[0] - 1, nrows))

    return grid_x, grid_y
