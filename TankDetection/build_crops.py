import argparse
import os
import glob
import detect_tanks as dt
import rasterio
import multiprocessing

#from __future__ import print_function

import sys
import multiprocessing
import multiprocessing.pool
import traceback
from time import time


def main(folder, output_fold, iso_th,zoom, low,high, save_res):
    begin = time()
    nb_workers=multiprocessing.cpu_count()

    files = glob.glob(os.path.join(folder, '*.jp2'))
    
    file_by_date = {}
    k = 0
    for f in files:
        key = os.path.basename(f)[:-8]
        if key in file_by_date.keys():
            file_by_date[key] = file_by_date[key] + [f]
        else:
            file_by_date[key] = [f]
    for lista in list(file_by_date.values()):
        if len(lista) == 4:
            with rasterio.open(lista[0]) as src:
                w = src.width
                h = src.height
                print('w: ', w)
                print('h: ', h)
                for x_left in range(0, w, 1000):
                    for y_left in range(0, h, 1000):
                        B02 = lista[0][:-8]+'_B02.jp2'
                        B03 = lista[0][:-8]+'_B03.jp2'
                        B04 = lista[0][:-8]+'_B04.jp2'
                        B08 = lista[0][:-8]+'_B08.jp2'
                        dt.detect_tanks(B02, B03, B04, B08, x_left, y_left, 1000, 1000, iso_th=iso_th,low=low,high=high,zoom=zoom,fold_path=output_fold,save_res_img=save_res)
                        k+=1
    end  = time()-begin
    print("full processing time:", end)
    print("average processing time:", end/k)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('run multimage tank detection pipeline'))
    parser.add_argument('--input', help=('folder that all the input images are stored'))
    parser.add_argument('--output', help=("output path"), type=str, default="./")
    parser.add_argument('--iso_th', help=('isoperimetric threshold'),default=0.9, type=float)
    parser.add_argument('--zoom', help=('apply 2x fft_zoom'),default=False, action='store_true')
    parser.add_argument('--save_res', help=('save result images'),default=False, action='store_true')
    parser.add_argument('-lth','--low', help=('low threshold for Canny-Devernay'),default=400, type=int)
    parser.add_argument('-hth','--high', help=('high threshold for Canny-Devernay'),default=700, type=int)
    args = parser.parse_args()
    main(args.input, args.output, args.iso_th, args.zoom, args.low, args.high,args.save_res)
