#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:51:50 2019

@author: antoine
"""


import numpy as np 
import cv2 
import os 
import json
import matplotlib.pyplot as plt 
import scipy.misc
import scipy.ndimage
import skimage.morphology
from skimage.morphology import disk
from skimage.draw import rectangle, rectangle_perimeter


import warnings
import pyproj
import rasterio

        
"""
fonction qui cherche à determiner si un segment fermé est un cercle, en regardant sont rapport d'isopérimétrie, 
le seuil doit être inférieur à 1 (égalité pour les cercles parfaits)
"""


def get_list_of_edge(txt,closed=True):
    """
    Load list of edges

    Parameters
    ----------
    txt : str
        Path to list of edges.
    closed : Bool, optional
        Set to True to keep only closed segment. The default is True.

    Returns
    -------
    list
        list of segment.

    """
    coord = np.loadtxt(txt)
    if len(coord.shape)==1:
        return None
    list_of_segment = np.split(coord,np.argwhere(coord[:,0]<0).reshape(-1))
    list_of_segment = [x[1:,:] if x[0,0] <0 else x for x in list_of_segment[:-1]]
    list_of_segment= [x for x in list_of_segment if x.shape[0]>0]
    if closed:
        closed_edge_list = [segment for segment in list_of_segment if is_closed(segment)]
        return closed_edge_list
    else:
        return list_of_segment
    

def stretch_im(im_np):
    im_np = im_np.astype(int)
    im_np *= 3
    im_np[im_np > 255] = 255
    return im_np
    

def tophat(im_np):
    tophat_np = []
    if len(im_np.shape) >= 3: 
        for i in range(im_np.shape[2]):
            tophat_np.append(im_np[:,:,i] - skimage.morphology.opening(im_np[:,:,i], selem=disk(10)))
    else : 
        tophat_np.append(im_np - skimage.morphology.opening(im_np, selem=disk(10))) 
    return np.max(tophat_np, axis=0)
    

def bottomhat(im_np):
    bottomhat_np = []
    if len(im_np.shape) >= 3: 
        for i in range(im_np.shape[2]):
            bottomhat_np.append(skimage.morphology.closing(im_np[:,:,i], selem=disk(10)) - im_np[:,:,i])
    else : 
        bottomhat_np.append(skimage.morphology.closing(im_np, selem=disk(10)) - im_np[:,:,i])
    return np.max(bottomhat_np, axis=0)





def read_json(json_file):
    with open(json_file) as f :
        data = json.load(f)
    
    return data

def to_255_pxl(img):
    """
    Return an image with pixel in [0,255]

    Parameters
    ----------
    img : numpy array
        Input image.

    Returns
    -------
    numpy array
        Output image.

    """
    """
    Retourne une image dont les pixels sont entre 0 et 255
    """
    img = img.astype(float)
    return np.round(255*(img - img.min(axis=(0,1)))/(img.max(axis=(0,1))-img.min(axis=(0,1)))).astype(int)



def PolyArea(x,y):
    """
    Computing area using shoalace formula

    Parameters
    ----------
    x : numpy array
        List of x coordinates.
    y : numpy array
        List of y coordinates.

    Returns
    -------
    float
        Area of a closed segment.

    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



def PolyPerimeter(coord):
    """
    Compute perimeter

    Parameters
    ----------
    coord : numpy array
        List of edge point coordinate.

    Returns
    -------
    p : float
        Perimeter.

    """
    coord_roll = np.roll(coord,shift=1, axis=0)
    p = np.sum(np.sqrt(np.sum((coord - coord_roll)**2,axis=1)))
    return p





def is_closed(segment): 

    """
    segment: numpy array
        a detected edge segment.
    
    Returns
    -------
     Bool
          True if the segment is closed, False if not.
    """
    x_o, y_o = segment[0,:] 
    x_f, y_f = segment[-1,:] 
    
    if (x_o == x_f) and (y_o == y_f):
        return True
    else: 
        return False
    
   

def select_circles(list_of_edge,threshold=0.8):
    
    circles = []
    if list_of_edge is None :
        return circles
    for i in range(len(list_of_edge)):
        segment = list_of_edge[i]
        p = PolyPerimeter(segment[:,:])
        x,y = segment[:,1], segment[:,0]  # correction de l'inversion des coordonnées, on ne prend pas le dernier terme
        a = PolyArea(x,y)
        
        q = 4*np.pi*a/(p**2)
#        print('q is equal to ', q)
        m_x, m_y = np.mean(x), np.mean(y)
#        radius = (np.max(x)-np.min(x) + np.max(y) - np.min(y))/4
        radius = np.sqrt(a/np.pi)
        if q > 1 :
            print(" Warning !!! q is greater than 1 at {} !!!".format(i))
        if q >= threshold and q <=1 :
            circles.append((m_x,m_y,radius))
        
    return circles

def compute_iso_ratio(list_of_edge):
    
    res = []
    if list_of_edge is None :
        return res
    for i in range(len(list_of_edge)):
        segment = list_of_edge[i]
        p = PolyPerimeter(segment[:,:])
        x,y = segment[:,1], segment[:,0]  # correction de l'inversion des coordonnées, on ne prend pas le dernier terme
        a = PolyArea(x,y)
        
        q = 4*np.pi*a/(p**2)
#        print('q is equal to ', q)
        m_x, m_y = np.mean(x), np.mean(y)
#        radius = (np.max(x)-np.min(x) + np.max(y) - np.min(y))/4
        radius = np.sqrt(a/np.pi)
        if q > 1 :
            print(" Warning !!! q is greater than 1 at {} !!!".format(i))
        else:
            res.append((segment,q))
        
    return res
        
   
def get_ground_truth(img,data):
    """
    input: 
        img: np.array, image
        data: dict, json correspondant à l'image
    """
    gt = data['annotations'] # on extrait les données de détections
    output = img.copy()
    mask = np.zeros_like(output)[:,:,0]
    
    for tank_data in gt:
        bbox = tank_data['bbox']
        rx, ry = rectangle(start=(bbox[1], bbox[0]),extent=(bbox[2], bbox[3]))
        rpx, rpy = rectangle_perimeter(start=(bbox[1], bbox[0]),extent=(bbox[2], bbox[3]))
        output[rpx,rpy] = (250,200,20)
        mask[np.int64(rx),np.int64(ry)] = 1
        
    nb_tanks =len(list(gt))
    
    return output, mask, nb_tanks 


def get_bbox_mask(data,N=513):
    """
    input: 
        img: np.array, image
        data: dict, json correspondant à l'image
    """
    
    try : 
        gt = data['annotations'] # on extrait les données de détections
        N = gt[0]['height'] 
    except : 
        gt = []
        
    mask = np.zeros((N,N))
    
    for tank_data in gt:
        bbox = tank_data['bbox']
        rx, ry = rectangle(start=(bbox[1], bbox[0]),extent=(bbox[2], bbox[3]),shape=(N,N))
        mask[np.int64(rx),np.int64(ry)] = 1
        
    nb_tanks =len(list(gt))
    
    return mask, nb_tanks 
        
        
    
def centers2mask(centers,size):
    """
    input : 
        centers: np.array (N,2) avec N le nombre de centres de tanks détectés
        size: tuple, taille de l'image (2D)
    """
    mask=np.zeros(size)
    for i in range(centers.shape[0]):
        y, x = centers[i]
        mask[int(y), int(x)] = True
    
    return mask


    
    
    
def pyproj_transform(x, y, in_epsg, out_epsg):
    """
    Wrapper around pyproj to convert coordinates from an EPSG system to another.
​
    Args:
        x (scalar or array): x coordinate(s) of the point(s), expressed in `in_epsg`
        y (scalar or array): y coordinate(s) of the point(s), expressed in `in_epsg`
        in_epsg (int): EPSG code of the input coordinate system
        out_epsg (int): EPSG code of the output coordinate system
​
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
​
    Args:
        path_to_img (str): path or url to an orthorectified georeferenced image
        lon (scalar or array): longitude or list of longitudes
        lat (scalar or array): latitude or list of latitudes
​
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



#if __name__ == "__main__":
#    img = "2018-04-18_L8_OLITIRS_LC81770392018108LGN00_band_B8.tif"
#    lat, lon = 29.979085, 31.134169
#    print("Pixel coordinates: {}, {}".format(*lonlat_to_pix(img, lon, lat)))