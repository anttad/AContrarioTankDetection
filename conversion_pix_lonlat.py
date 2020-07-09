#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:30:30 2020

@author: antoine
"""

import warnings
import pyproj
import rasterio


def pyproj_transform(x, y, in_epsg, out_epsg):
    """
    Wrapper around pyproj to convert coordinates from an EPSG system to another.

    Args:
        x (scalar or array): x coordinate(s) of the point(s), expressed in `in_epsg`
        y (scalar or array): y coordinate(s) of the point(s), expressed in `in_epsg`
        in_epsg (int): EPSG code of the input coordinate system
        out_epsg (int): EPSG code of the output coordinate system

    Returns:
        scalar or array: x coordinate(s) of the point(s) in the output EPSG system
        scalar or array: y coordinate(s) of the point(s) in the output EPSG system
    """
    with warnings.catch_warnings():  # noisy warnings here with pyproj>=2.4.2
        warnings.filterwarnings("ignore", category=FutureWarning)
        in_proj = pyproj.Proj(init="epsg:{}".format(in_epsg))
        out_proj = pyproj.Proj(init="epsg:{}".format(out_epsg))
    return pyproj.transform(in_proj, out_proj, x, y)


def lonlat_to_pix(path_to_img, lon, lat):
    """
    Convert geographic (lon, lat) coordinates to pixel (col, row) coordinates.

    Args:
        path_to_img (str): path or url to an orthorectified georeferenced image
        lon (scalar or array): longitude or list of longitudes
        lat (scalar or array): latitude or list of latitudes

    Returns:
        c (scalar or array): pixel(s) column coordinate(s)
        r (scalar or array): pixel(s) row coordinate(s)
    """
    with rasterio.open(path_to_img) as f:
        crs = f.crs
        transform = f.transform

    epsg = int(crs.data["init"].split(":")[1])
    x, y = pyproj_transform(lon, lat, 4326, epsg)
    return ~f.transform * (x, y)


def pix_to_lon_lat(path_to_img, x, y):
    """
    Convert geographic (lon, lat) coordinates to pixel (col, row) coordinates.
​
    Args:
        path_to_img (str): path or url to an orthorectified georeferenced image
        x (uint): col number in the image
        y (uint): row number in the image
​
    Returns:
        lat (scalar or array)
        lon (scalar or array)
    """
    with rasterio.open(path_to_img) as f:
        crs = f.crs
        transform = f.transform
    
    crs_x, crs_y = transform * (x, y)
    epsg = int(crs.data["init"].split(":")[1])
    lon, lat = pyproj_transform(crs_x, crs_y, epsg, 4326)
    return lat, lon


if __name__ == "__main__":
    img = "/home/antoine/Documents/THESE_CMLA/TankDetection/tanks/test_araucaria_17/2019-06-04_S2A_orbit_038_tile_22JFS_L1C_band_B02.tif"
    lat, lon = 29.979085, 31.134169
    x,y = lonlat_to_pix(img, lon, lat)
    print("Pixel coordinates: {}, {}".format(x,y))
    new_lat, new_lon = pix_to_lon_lat(img, x, y)

    print("lat, lon coordinates: {}, {}".format(new_lat, new_lon))