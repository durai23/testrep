#!/bin/bash

EXE="module purge --force"
echo $EXE
eval "$EXE"

EXE="module use /usr/local/software/jureca/OtherStages"
echo $EXE
eval "$EXE"

echo 

EXE="module load Stages/2016b"
echo $EXE
eval "$EXE"
EXE="module load GCC/5.4.0"
echo $EXE
eval "$EXE"
EXE="module load Python/2.7.12"
echo $EXE
eval "$EXE"
EXE="module load FSL/5.0.9"
echo $EXE
eval "$EXE"
EXE="source ${FSLDIR}/etc/fslconf/fsl.sh"
echo $EXE
eval "$EXE"
# EXE="module load FreeSurfer/5.3.0-centos6_x86_64"
# echo $EXE
# eval "$EXE"
#EXE="module load Atom/1.14.2"
#echo $EXE
#eval "$EXE"
EXE="module load ParaStationMPI/5.1.5-1 mpi4py/2.0.0-Python-2.7.12"
echo $EXE
eval "$EXE"

echo

EXE="module use /data/inm1/mapping/software/2016b/modules"
echo $EXE
eval "$EXE"

echo

EXE="module load FSL_extra/5.0.9"
echo $EXE
eval "$EXE"
EXE="module load MRtrix/0.3.15"
echo $EXE
eval "$EXE"
EXE="module load ANTs/2.1.0"
echo $EXE
eval "$EXE"
EXE="module load ANTs_extra/2.1.0"
echo $EXE
eval "$EXE"
EXE="module load FZJ/1.0.0"
echo $EXE
eval "$EXE"

#EXE="module load FZJ_dMRI/1.0.0"
#echo $EXE
#eval "$EXE"
