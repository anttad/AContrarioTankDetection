#!/bin/bash

# usage
if [ "$#" -lt 2 ]
then
  echo "usage: my_processing.sh IN_IMG OUT_DIR"
  exit 1
fi

# In dir est le dossier contenant la position "lat/lon" à traiter, chaque canal contient une bande spectrale à un instant donné. 
IN=$1
OUT=$2
LTH=$3
HTH=$4
# create output directory
mkdir ${OUT}

# external Ghostscript path
GS=`which gs`

# process each image
#for i in ${IN}/*pca.tif
#do
echo ------------------------------------
n=`basename ${IN} .tif`
echo image: ${IN}

  # convert image to ASC format, including all channels and a graylevel version
./C_exe/image2asc ${IN} ${OUT}/$n.asc
  #./image2asc_gray_0.1/image2asc_gray $i ${OUT}/${n}_gray.asc

  # extract channels
  #./image_channels_0.1/image_channels ${OUT}/$n.asc

  # print image size
echo -n "image size: "
head -n 1 ${OUT}/$n.asc

  # process channels 2, 3, 4 and 8 (1, 2, 3 and 7, when count starts at zero)
  rm -f ${OUT}/${n}_detection.txt
  # for j in ${OUT}/${n}*_channel1.asc ${OUT}/${n}*_channel2.asc ${OUT}/${n}*_channel3.asc ${OUT}/${n}*_channel7.asc
  j=${OUT}/$n.asc
  
  echo "  channel: $j"
#for s in 0 #0.1 #0.2 0.3 #0.5 0.7 1 # ecart-type du filtre gaussien, voir si on garde tout où si on supprime certaine variance (ex: prendre 0 0.1 0.2 0.3)
#do
echo -n "    blur 0 computing time(s): "
TIMEFORMAT=%R

    # compute curves using Devernay algorithm
    # a modified version is used which gives at the output only
    # closed curves, roughly circular with a given range of radius
    # time ./devernay_1.0_closed_given_radius/devernay $j -t ${OUT}/dv.txt -s $s # TODO: changer et utiliser le devernay classique
time ./C_exe/devernay $j -t ${OUT}/dv.txt -s 0 -l $LTH -h $HTH -p ${OUT}/${n}_detection.pdf
cat ${OUT}/dv.txt >> ${OUT}/${n}_detection.txt
#done


  # count number of detected curves
  #grep "\-1 \-1" ${OUT}/${n}_detection.txt | wc -l > ${OUT}/${n}_detection_num.txt
  #echo total number of curves detected: `cat ${OUT}/${n}_detection_num.txt`
  

  # add an end-of-curve code so image_curves2eps can handle empty files
  # (this must be done AFTER counting number of detected curves, otherwise
  #  the count will be increased by one)
echo "-1 -1" >> ${OUT}/${n}_detection.txt

  # generate vectorial PDF output including image and curves
  #./image_curves2eps_0.1/image_curves2eps ${OUT}/${n}_gray.asc ${OUT}/${n}_detection.txt 1.5 0.75 > ${OUT}/${n}_detection.eps
  #$GS -sDEVICE=pdfwrite -dEPSCrop -dPDFSETTINGS=/prepress -o ${OUT}/${n}_detection.pdf ${OUT}/${n}_detection.eps > /dev/null
  # remove auxiliary files
  # rm ${OUT}/${n}*asc
  # rm ${OUT}/${n}*eps
rm ${OUT}/dv.txt 

