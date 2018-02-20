#!/usr/bin/env bash

INPUTDIR=$1
PREFIX=$2
OUTDIR=$3


TMP=${SBIA_TMPDIR}/${PREFIX}

if [ -d ${TMP} ]
then
	rm -rfv ${TMP}
fi

mkdir ${TMP}
cp ${INPUTDIR}/* ${TMP}
file=`ls -1 ${INPUTDIR}/*`
ext=${file##*.}
if [ "${ext}" = "bz2" ]
then
	bunzip2 ${TMP}/*
fi

dcm2nii -r N -o ${OUTDIR} ${TMP}/*
f=`ls -1 ${OUTDIR}/*.nii.gz`
if [ ! -e ${f} ]
then
	echo "Conversion Failed for ${PREFIX} "
	exit
fi

mv ${f} ${OUTDIR}/${PREFIX}.nii.gz

rm -fv ${TMP}/*
rmdir ${TMP}

	
