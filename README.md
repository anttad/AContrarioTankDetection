# A Contrario Tank Detection


## Methodology: 

The méthode is based on the IGARSS paper *Oil tank detection in satellite images via a contrario clustering*.

From the B02, B03, B04 and B08 bands, a greyscale image is computed via PCA.
Then the Canny-Devernay procedure is applied to detect closed edge segments on which an isoperimetric threshold is applied.

This gives a first detection of tanks which are then clustered by using binary dilation of varius radii. Finally, the clusters are filtered based on their NFA score.

## Code

To use the code, you must first compile the Canny-Devernay edge detector code:

```
mkdir build
cd build
cmake ../devernay1.1/
make
```

The binary files will be moved to *TankDetection/C_exe*.


The detector function is named `detect_tank` and is in *TankDetection/detect_tanks.py*.
The file *TankDetection/build_crops.py* contains a wrapper for sentinel-2  .jp2 images. 

The options for this wrapper are the following : 

```
  -h, --help                	show this help message and exit
      --input INPUT         	folder that all the input images are stored
      --output OUTPUT       	output path
      --iso_th ISO_TH       	isoperimetric threshold
      --zoom                	apply 2x fft_zoom
      --save_res            	save result images  
      --auto_th             	use Otsu's method to set Canny-Devernay's threshold
  -lth LOW, --low LOW  			low threshold for Canny-Devernay
  -hth HIGH, --high HIGH 		high threshold for Canny-Devernay
  -cw CROP_W, --crop_w CROP_W 	Crop width
  -ch CROP_H, --crop_h CROP_H  	Crop height
```

Usage : 

```
python build_crops.py --input ./im_test --output ./test_result --save_res --auto_th --zoom
```

`build_crops` will partition the input image (**.jp2** or **.tif** files)  according to the `-cw` and `-ch` options. Each crop will then be processed to detect tank farm and the results will be stored in the folder specified by the `--output` argument, were a folder will be named after the chosen parameters. In this file you will find a folder for each crop with the gray_scale image, the edge map, and the json file where the detections are saved.
`--save_res` will save the image result for analysis.






