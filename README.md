# AContrarioTankDetection
Code for IGARSS 2020 paper  "Oil tank detection in satellite images via a contrario clustering"




First, unzip "oil_tank_detection_0.1.zip" and compile the C code (follow the README instructions)

Then, copy the "devernay" and "image2asc" executable that have been build to TankDetection/C_exe

finaly, to run the method in python use the function "detect_tanks(...)" in "detect_tanks.py" 

An image of the detection can be produced by setting "save_res_img" to True.

The "build_crop.py" is a wraper for the sentinel-2  images .jp2 dowloaded from sentinel-hub. B02, B03, B04 and B08 channels are used by the method.
