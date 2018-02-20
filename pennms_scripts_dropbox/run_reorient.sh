#!/usr/bin/env bash

for i in ` cat ~/Brain_Tumor_2015/Data/subsToProcess.csv | cut -d, -f1 | grep -v "ID"`
do 
	for j in t1 t2 flair t1ce perf
	do 
		f=`ls -1 /sbia/sbiaprj/brain_tumor/Brain_Tumor_2015/Data/${i}/${i}0/Nifti/${i}0_${j}/${i}0_${j}.nii.gz` 
		if [ -z ${f} ]
		then
			continue
		fi

		OUTDIR=/sbia/sbiaprj/brain_tumor/Brain_Tumor_2015/Protocols/ReOriented/${i}/${i}0/${i}0_${j}
		echo ${OUTDIR}
		echo ${OUTDIR}/${i}0_${j}_LPS.nii.gz

		if [ ! -d ${OUTDIR} ]
		then
			mkdir -p ${OUTDIR}
		fi

		qsub -l short -j y -o ${OUTDIR}/\$JOB_NAME-\$JOB_ID-${i}0_${j}.log \
		/sbia/sbiaprj/brain_tumor/Brain_Tumor_2015/Scripts/ClearSFORM+ReorientLPS.sh \
		${f} \
		${OUTDIR}/${i}0_${j}_LPS.nii.gz
	done
done
