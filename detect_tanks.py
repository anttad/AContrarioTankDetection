#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from conversion_pix_lonlat import pix_to_lon_lat
import os
import numpy as np 
import cv2
import json 

from skimage import  color #, data 
from skimage.transform import hough_circle, hough_circle_peaks,hough_ellipse 
from skimage.feature import canny
from skimage.draw import circle_perimeter, rectangle_perimeter,rectangle,circle, polygon_perimeter
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu

from skimage.morphology import binary_closing, binary_dilation
from skimage.morphology import square

import scipy.misc
import scipy.ndimage as ndimage
import skimage.morphology
from skimage.morphology import disk
from skimage.viewer import ImageViewer
import skimage.io
#import argparse
from scipy.stats import binom
from sklearn.decomposition import PCA

from utils import * 
#from metrics import true_positive_rate
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from time import time


def fftzoom(img, factor=2):
    """
    Apply fftzoom.
    
    Parameters
    
    ----------
    img : numpy array
        Input image.
    factor : int , optional
        zoom factor. The default is 2.
        
    Returns
    -------
    res : numpy array
        zoomed image
        
    """
    
    if len(img.shape) == 2:
        nrow,ncol = img.shape
        r_alpha, c_alpha= nrow*(factor-1),ncol*(factor-1)
        r_step, c_step = int(np.ceil(r_alpha/2)), int(np.ceil(c_alpha/2))
        fft_im = np.fft.fftshift(np.fft.fft2(img))
        fft_pad = np.pad(fft_im,((r_step,r_step), (c_step, c_step)))
        res = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_pad)))
#        res = np.uint8(255*(res-res.min())/(res.max()-res.min()))
        
    elif len(img.shape)==3:
        nrow,ncol,_ = img.shape
        r_alpha, c_alpha= nrow*(factor-1),ncol*(factor-1)
        r_step, c_step = int(np.ceil(r_alpha/2)), int(np.ceil(c_alpha/2))
        res_list =[]
        for i in range(3):
            fft_im = np.fft.fftshift(np.fft.fft2(img[:,:,i]))
            fft_pad = np.pad(fft_im,((r_step,r_step), (c_step, c_step)))
            res_temp = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_pad)))
            per_inf = np.percentile(res_temp,q=1)
            per_sup = np.percentile(res_temp,q=99)
            res_temp= np.clip(res_temp,per_inf,per_sup)
            res_temp = np.uint8(255*(res_temp-res_temp.min())/(res_temp.max()-res_temp.min()))
            res_list.append(res_temp)
        res= np.stack(res_list, axis=2)
#        res = np.uint8(255*(res-res.min())/(res.max()-res.min()))
        
    return res



        

def computelogNFA( N,k,sigma,Ntest=2500):
    
    logNFA = -(np.log(Ntest) + binom.logsf(k-1,N,sigma)) # k-1 pour éviter l'inégalité stricte
    
    return logNFA
            
            

            


#    edge_listdir

def channel_reduction_pca(img,stretch_dyn=False,dtype='uint16'):
    """
    
    Get greyscale image via pca reduction.
    
    Parameters
    ----------
    img : numpy array
        Image with all its channels.
    stretch_dyn : Bool, optional
        Set to True to improve contrast. The default is True.
        
    Returns
    -------
    new_img : numpy array
        Greyscale image.
    
    """
    pca = PCA(n_components=1)
    
    chan_list = [img[:,:,i].flatten() for i in range(img.shape[-1])]
    chan_arr = np.array(chan_list).T
    new_img= pca.fit_transform(chan_arr).reshape(img.shape[0],img.shape[1])
    print('new_img range:', new_img.min(), new_img.max())
    new_img -= new_img.min()
    if stretch_dyn : 
        if dtype=='uint16':
            new_img = np.uint16(np.round(normalize(new_img)*(2**16 -1)))
        elif dtype == 'uint8':
            new_img = np.uint16(np.round(normalize(new_img)*(2**8 -1)))
    print('new_img range:', new_img.min(), new_img.max())
    return new_img
    
    

def normalize(img):
    
    return (img - img.min())/(img.max() - img.min()+0.01)


def img_dyn_enhancement(img,q_inf=1,q_sup=99,bit_16=True):
    
    """
    Enhance image dynamique.
    
    Parameters
    ----------
    img : numpy array
        Input image.
    q_inf : int, optional
        Lower percentile. The default is 1.
    q_sup : int, optional
        Higest percentile. The default is 99.
    bit_16 : Bool, optional
        Set to True if you want the output to be a 16 bit image. The default is True.
        
    Returns
    -------
    img : numpy array
        enhance image.
    """
    
    if len(img.shape)==3:
        for i in range(img.shape[-1]):
            per_inf = np.percentile(img[:,:,i],q=q_inf)
            per_sup = np.percentile(img[:,:,i],q=q_sup)
            new_chan =np.clip(img[:,:,i],per_inf,per_sup)
            img[:,:,i] = (new_chan-new_chan.min())/(new_chan.max()-new_chan.min()) 
        if bit_16:
            img = np.uint16((2**16-1)*img)
        else : 
            img = np.uint8((2**8-1)*img)
    return img
 
            
def process_clustering(centers, labels,epsilon=1):
    
    """
    Process the clusters obtained through binary dilations
    
    Parameters
    ----------
    centers : list of 2D tuples.
        List of the centers of detected circles.
    labels : numpy array
        Cluster labels as an image.
    epsilon : float, optional
        NFA threshold. The default is 1.
        
    Returns
    -------
    clusters_list: list 
        List of clusters considered as meaningful in the a contrario framework
        
    nfa_list: list
        list of NFA associated with each clusters
    
    """
    log_eps = -np.log(epsilon)
    N = len(centers) #number of detected circles
#    centers_label = np.zeros((n_c,))
    n_labels = len(np.unique(labels)) - 1
    clusters = dict()
    aire_image= labels.shape[0]*labels.shape[1]
    clusters_list = []
    nfa_list = []
    for label in range(1,n_labels+1):
        mask = labels == label
        clusters[label] = {'logNFA':None, 'centers':[], 'centers_id':[],'rectangle': None,'sigma':None,'meaningful':None} 
        for i in range(N):
            cy,cx,_ = centers[i]
            max_y, max_y= mask.shape
#            if int(c_y)
            if mask[int(cy),int(cx)]:
                clusters[label]['centers'].append(np.array([cy,cx]))
                clusters[label]['centers_id'].append(i)
        
        cl_pts = clusters[label]['centers']
        k = len(cl_pts)
        if k <= 2:
            
            cl_pts_arr = np.array(cl_pts)
            mu_g = cl_pts_arr.mean(axis=0)
            ly, lx =  np.min(cl_pts - mu_g,axis=0)
            ry, rx =  np.max(cl_pts - mu_g,axis=0)
            pts1, pts2= mu_g + np.array([ly,lx]), mu_g + np.array([ry,rx])
            
            clusters[label]['sigma']= 0
            clusters[label]['rectangle']= (pts1,pts2)
            clusters[label]['logNFA']= -np.inf
        
        else : 
            cl_pts_arr = np.array(cl_pts)
            mu_g = cl_pts_arr.mean(axis=0)
            ly, lx =  np.min(cl_pts - mu_g,axis=0)
            ry, rx =  np.max(cl_pts - mu_g,axis=0)
            pts1, pts2= mu_g + np.array([ly,lx]), mu_g + np.array([ry,rx])
            sigma = (ry-ly)*(rx-lx)/aire_image
            logNFA = computelogNFA(N,k,sigma)
            clusters[label]['sigma']=sigma
            clusters[label]['rectangle']= (pts1,pts2)
            clusters[label]['logNFA']= logNFA
            clusters[label]['meaningful'] = logNFA > log_eps
            clusters_list.append(clusters[label])
            nfa_list.append(logNFA)      
    return clusters_list, nfa_list        
            
        
def multiple_clusering(mask,circles,min_rad=1,max_rad=10, epsilon=1):
    
    """
    Run the clustering on multiple binary dilation radii.
    
    Parameters
    ----------
    mask : numpy array
        Mask of circles detection.
    circles : list
        List of circles centers coordinates.
    min_rad : int, optional
        Smallest radius. The default is 1.
    max_rad : int, optional
        Biggest radius. The default is 10.
    epsilon : float, optional
        NFA threshold. The default is 1.
        
    Returns
    -------
    clusters : list
        list of clusters.
    nfa_list : list
        list of NFA.
        
    """
    
    clusters = []
    nfa_list=[]
    for rad in range(min_rad,max_rad):
        
        mask_closed = binary_dilation(mask,disk(rad))
        labels, n_object = ndimage.label(mask_closed)
        cl, nfal = process_clustering(circles,labels,epsilon=epsilon)
        clusters += cl
        nfa_list += nfal
    return clusters,nfa_list


def cluster_filtering(clusters,circles, nfa_list,epsilon=1):
    
    """
    Apply the exclusion princple for the overlapping clusters.
    
    Parameters
    ----------
    clusters : list 
        list of clusters.
    circles : list
        list of circle centers coordinates.
    nfa_list : list
        List of NFA measure for the clusters.
    epsilon : float, optional
        NFA threshold. The default is 1.
    
    Returns
    -------
    clusters : list
        list of selected clusters.
    """
    
    log_eps = -np.log(epsilon)
    if len(clusters)<=1:
        return clusters
    else :
        nfa_argsort = np.argsort(nfa_list)
        #
        N = len(circles) # on compte le nombre de points dans le cluster
        #
        clusters_to_keep = [nfa_argsort[-1]] # list des clusters à garder, on commence par le dernier
    #    nfa_to_keep = [nfa_list[nfa_argsort[-1]]] # NFA des clusters gardés
                    #l2 = tree.nodes[clusters_to_keep[0]].points_id 
                    #print(l2)
        nfa_argsort=nfa_argsort[:-1]
        #removed_pts = [pt for pt in clusters[clusters_to_keep[0]]['centers'] ]
        
        removed_pts=[]
        
        removed_pts += clusters[clusters_to_keep[-1]]['centers_id']  # on stock les points qui ne peuvent plus être comptés
        
        while len(nfa_argsort) >0:
                        
            new_best_nfa = -np.inf
            best_clust_id = None
            clust_to_rm = []
    #        print("\nnfa argsort", nfa_argsort)
        #    print("removed_pts", removed_pts )
            for ind in nfa_argsort:
                l1 = clusters[ind]['centers_id']
                new_points = [pt for pt in l1 if not pt in removed_pts]
                k = len(new_points)
                if k <= 1:
                    clusters[ind]['meaningful'] = False
                    clust_to_rm.append(ind)
                else: 
                    sigma = clusters[ind]['sigma']
                    logNFA = computelogNFA( N,k,sigma,Ntest=2500)
    #                print("k, logNFA, bestNFA\n", k, logNFA, new_best_nfa)
                    if logNFA < log_eps:
                        clusters[ind]['meaningful'] = False
                        clust_to_rm.append(ind)
                    if logNFA > new_best_nfa:
                        new_best_nfa = logNFA
                        best_clust_id =ind
                        rm_pts = new_points
            
            if len(clust_to_rm)!=0 :
                nfa_argsort=np.array([ind for ind in nfa_argsort if ind not in clust_to_rm])     
            
            if new_best_nfa > log_eps :
                clusters_to_keep.append(best_clust_id)
                nfa_argsort = nfa_argsort[nfa_argsort!=best_clust_id]
                removed_pts = removed_pts + rm_pts
        return clusters
            
    
    
#def detect_tanks(im_path, meta_tif=None, save_res_img=False):
def detect_tanks(B02, B03, B04, B08, window_x0=0, window_y0=0, w=1000, h=1000, save_res_img=False,iso_th=0.9,low=400,high=700,zoom=False,fold_path='./',auto_th=False):    
    
    """
    Run tank detection on a the B02, B03, B04, and B08 channels of a Sentinel-2 image.
    
    Parameters
    ----------
    B02 : str
        Path to B02 file.
    B03 : str
        Path to B03 file.
    B04 : str
        Path to B04 file.
    B08 : str
        Path to B08 file.
        
    window_x0 : int, optional
        Window of the crop. The default is 0.
    window_y0 : int, optional
        Window of the crop. The default is 0.
    w : int, optional
        image width. The default is 1000.
    h : int, optional
        Image height. The default is 1000.
    save_res_img : Bool, optional
        Set to True to save the output as images. The default is False.
    iso_th : float, optional
        Isoperimetric threshold (between 0 and 1). The default is 0.9.
    low : int, optional
        Low threshold from Canny-Devernay's method. The default is 400.
    high : int, optional
        High threshold from Canny-Devernay's method. The default is 700.
    zoom : Bool, optional
        Set to True to apply fftzoom. The default is False.
    fold_path : str, optional
        Path to the folder where to store the results. The default is './'.
    
    Returns
    -------
    None.
    """
    
    
    process_time = time()
    c_path = "./build"
    devernay = os.path.join(c_path,"devernay")
    
    if zoom:
        low,high = int(low/2), int(high/2)
        
   
    im_name = os.path.basename(B02)[:-8] + '_x_'+str(window_x0)+'_y_'+str(window_y0)
    print('high value: ', high,'\n')
    #res_folder = os.path.join(os.path.dirname(im_path),'res_'+im_name)
    #res_folder = os.path.join('/home/l.carvalho/data/tanks_filipinas')
#    par_res_fold = "/home/antoine/Documents/Lucas_Code/tank_detection_pipeline/TankDetection/res_iso_{}_l_{}_h_{}_zoom_{}".format(iso_th,low,high,zoom)
#    par_res_folder = '/home/l.carvalho/data/tanks_filipinas/res_iso_{}_l_{}_h_{}_zoom_{}'.format(iso_th,low,high,zoom)
    fold_name = "res_iso_{}_l_{}_h_{}_zoom_{}_auto_th_{}".format(iso_th,low,high,zoom,auto_th)
    par_res_fold = os.path.join(fold_path,fold_name)
    os.system('mkdir {}'.format(par_res_fold))
    res_folder = os.path.join(par_res_fold,im_name)
    try: 
        os.mkdir(res_folder)
    except:
        print('folder already exist')
    #os.system('mkdir {}'.format(res_folder))
    
    #if os.path.exists(os.path.join(res_folder, im_name+'_pca.tif')):
        

    #img=skimage.io.imread(im_path)
    print('b02: ', B02, 'w: ', window_x0, 'h: ', window_y0)
    win = Window(window_x0, window_y0, w, h)
    
    try:
        with rasterio.open(B02) as src:
            b02_img = src.read(1, window=win)
        with rasterio.open(B03) as src:
            b03_img = src.read(1, window=win)
        with rasterio.open(B04) as src:
            b04_img = src.read(1, window=win)
        with rasterio.open(B08) as src:
            b08_img = src.read(1, window=win)
    except:
        print('image corrupted')
        return
    w_aux, h_aux = b02_img.shape
    img = np.zeros((w_aux, h_aux, 4))
    img[:,:,2] = b02_img
    img[:,:,1] = b03_img
    img[:,:,0] = b04_img
    img[:,:,3] = b08_img
    print("img dtype",b02_img.dtype)
    print("PCA reduction...")
    pca_time = time()
    img_pca = channel_reduction_pca(img,dtype=b02_img.dtype)
    if zoom:
        print("fft zoom...")
        zoom_time=time()
        img_pca = fftzoom(img_pca)
        print("fft zoom done in {:.2f}s".format(time()-zoom_time))
        
    im_pca_path = os.path.join(res_folder, im_name+'_pca.tif')
    print("PCA reduction done in {:.2f}s".format(time()-pca_time))
    skimage.io.imsave(im_pca_path, img_pca)
    
    
    epsilon = 0.01
    edge_folder=os.path.join(res_folder,"edges")
    try:
        os.mkdir(edge_folder)
    except:
        print('folder already exist')
    
    print('computing edges...')
    edges_time = time()
    print(os.getcwd())
    edge_file= im_name+'_pca_detection.txt'
    txt = os.path.join(edge_folder,edge_file)
    
    pdf_file= im_name+'_pca_detection.pdf'
    pdf = os.path.join(edge_folder,pdf_file)
    
    if auto_th:
        nbins=len(np.unique(img_pca.flatten()).tolist())
        print('nbins: ', nbins)
        high = min((threshold_otsu(img_pca,nbins),np.quantile(img_pca,0.35)))
        low = 0.5*high


    os.system("{} {}  -l {} -h {} -p {} -t {}".format(devernay, im_pca_path, low, high, pdf, txt)) 
#    os.system("./my_processing2.sh {} {} {} {}".format(im_pca_path,edge_folder, low, high)) 
    #circles = []
    print("Edge extraction done in {:.2f}s".format(time()-edges_time))
    print('             ')
    print('             ')
    print('im_name: ', im_name)
    print('res_folder: ', res_folder)
    print('im_pca_path: ', im_pca_path)
    print('edge_folder: ', edge_folder)
    print('im_name: ', im_name)
    print('txt', txt)
    print('auto_th: ', auto_th)
    print('low_th, high_th: ', low,high)
    print('             ')
    print('             ')
    
    print("\nComputing circles")
    circles_time=time()
    list_of_edges = get_list_of_edge(txt)
    circles = select_circles(list_of_edges,threshold=iso_th)
    print("Circles detected done in {:.2f}s".format(time()-circles_time))
    
    if save_res_img:
        seg_iso_per_rat = compute_iso_ratio(list_of_edges)


    im_dim = (img_pca.shape[0], img_pca.shape[1])
    output_shape=(img_pca.shape[0], img_pca.shape[1], 3)
    mask = np.zeros(im_dim)
    if save_res_img :
        output = np.zeros((img_pca.shape[0], img_pca.shape[1], 3))
        output[:,:,0],output[:,:,1],output[:,:,2] = np.uint8(np.round(255*normalize(img_pca))),np.uint8(np.round(255*normalize(img_pca))), np.uint8(np.round(255*normalize(img_pca)))
        output_pol = output.copy()
#    im_dim = output.shape

   
    for center_y, center_x, radius in circles:
        radius = max((2,radius))
        circy, circx = circle(int(center_y),int(center_x), int(radius), shape=mask.shape)
        mask[circy,circx] = 1
        if save_res_img :
            cpy, cpx = circle_perimeter(int(center_y),int(center_x), int(radius), shape=im_dim)
            output[cpy,cpx,:] = (0,255,255)
    
    print("clustering running ...")
    clust_time = time()
    if not zoom : 
        clusters, nfa_list= multiple_clusering(mask,circles,epsilon=epsilon)
    else : 
        clusters, nfa_list= multiple_clusering(mask,circles,epsilon=epsilon,max_rad=20)
    clusters=cluster_filtering(clusters,circles,nfa_list,epsilon=epsilon)
    print("Clusters computed in in {:.2f}s".format(time()-clust_time))

    json_dict = {}
    
    for cluster in clusters:     
        if cluster['meaningful']:
            
            rec1, rec2 = cluster['rectangle']
            recy, recx = rectangle_perimeter(rec1-2.5,rec2+2.5,shape=output_shape)
            rec1, rec2 = rec1-2.5,rec2+2.5
            if save_res_img:
                output[recy,recx] = (0,255,0)
            #if meta_tif is not None :
            #    lat1, lon1 = pix_to_lon_lat(meta_tif,rec1[1], rec1[0])
            #    lat2, lon2 = pix_to_lon_lat(meta_tif,rec2[1], rec2[0])
            #else :
            #    lat1, lon1 = pix_to_lon_lat(im_path,rec1[1], rec1[0])
            #    lat2, lon2 = pix_to_lon_lat(im_path,rec2[1], rec2[0] )   
            #json_dict[len(json_dict.keys())] = {"coordinates": [[lat1,lon1], [lat2,lon2]]}
            json_dict[len(json_dict.keys())] = {"coordinates": [[rec1[0], rec1[1]], [rec2[0], rec2[1]]]}

    json_file = im_name.split('.')[0]+'.json'
    json_path = os.path.join(res_folder,json_file)
    with open(json_path,'w') as f : 
        json.dump(json_dict,f)
        print("json saved")
    print('res_folder:', res_folder)
    
    end_process_time = time()-process_time
    print("full processing done in {:.2f}s".format(end_process_time))
    if save_res_img : 
        input_img = img[:,:,:-1]
        skimage.io.imsave(os.path.join(res_folder,'res.png'), np.uint8(output))
        output_fold = os.path.join(par_res_fold,'output')
        output_pol_fold = os.path.join(par_res_fold,'output_shapes')
        input_fold = os.path.join(par_res_fold,'input')
        os.system('mkdir {}'.format(output_fold))
        os.system('mkdir {}'.format(output_pol_fold))
        os.system('mkdir {}'.format(input_fold))
        skimage.io.imsave(os.path.join(output_fold,im_name+'.png'), np.uint8(output))
        skimage.io.imsave(os.path.join(input_fold,im_name+'.tif'), input_img)
        
        output_pol = np.uint8(255*((output_pol - output_pol.min())/(output_pol.max() - output_pol.min())))

        for seg, iso_rat in seg_iso_per_rat:
            
            x,y = seg[:,1], seg[:,0]
            rx, ry  = polygon_perimeter(x,y,output_pol.shape)
            if iso_rat >= iso_th :
                output_pol[rx,ry] = (0,255,0)
            else :
                output_pol[rx,ry] = (255,0,0)
#            
        skimage.io.imsave(os.path.join(output_pol_fold,im_name+'.png'), np.uint8(output_pol))

