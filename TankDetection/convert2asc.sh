#!/bin/bash

# usage
if [ "$#" -lt 2 ]
then
  echo "usage: process.sh IN_DIR OUT_DIR"
  exit 1
fi

# In dir est le dossier contenant la position "lat/lon" à traiter, chaque canal contient une bande spectrale à un instant donné. 
IN=$1
OUT=$2

# create output directory
mkdir ${OUT}

# external Ghostscript path
GS=`which gs`

# process each image
for i in ${IN}/*.tif
do
  echo ------------------------------------
  n=`basename $i .tif`
  echo image: $i

  # convert image to ASC format, including all channels and a graylevel version
  ./C_exe/image2asc $i ${OUT}/$n.asc
  #./image2asc_gray_0.1/image2asc_gray $i ${OUT}/${n}_gray.asc

  # extract channels
  #./image_channels_0.1/image_channels ${OUT}/$n.asc

  # print image size
  echo -n "image size: "
  head -n 1 ${OUT}/$n.asc
done
