#!/usr/bin/env bash

SUB=$1
INPUT=$2 #Input T1 N4 Image of same subject
REF=$3 #Reference T1 N4 Image
MASK=$4 
OUTDIR=$5

PID=$$
TMP=${SBIA_TMPDIR}/registerMask_${SUB}_${PID}
mkdir ${TMP}
echo "Using Temporary Directory ${TMP}"



#Run flirt with 6 dof
echo flirt -in ${REF} -ref ${INPUT} -out ${OUTDIR}/refTo_${SUB}.hdr -omat ${OUTDIR}/refTo_${SUB}.mat -v -dof 6
flirt -in ${REF} -ref ${INPUT} -out ${OUTDIR}/refTo_${SUB}.hdr -omat ${OUTDIR}/refTo_${SUB}.mat -v -dof 6 \
-searchrx -15 +15 \
-searchry -15 +15 \
-searchrz -15 +15

#apply flirt to mask
echo flirt -in ${MASK} -ref ${INPUT} -out ${OUTDIR}/refMaskTo_${SUB}.hdr -applyxfm -init ${OUTDIR}/refTo_${SUB}.mat -dof 6 -interp nearestneighbour
flirt -in ${MASK} -ref ${INPUT} -out ${OUTDIR}/refMaskTo_${SUB}.hdr -applyxfm -init ${OUTDIR}/refTo_${SUB}.mat -v -dof 6 -interp nearestneighbour


rm -rfv ${TMP}
