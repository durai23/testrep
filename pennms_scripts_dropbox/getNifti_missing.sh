#!/usr/bin/env bash

for i in AAKQ
do
	echo ${i}

	for j in `cat ~/Brain_Tumor_2015/Data/${i}/${i}0/dicoms/${i}0_dicomInfo_parsed.csv | grep -v "ID" | cut -d, -f1 | grep ^${i} ` 
	do
		
		PREFIX=`grep -w ^${j} ~/Brain_Tumor_2015/Data/${i}/${i}0/dicoms/${i}0_dicomInfo_parsed.csv  | cut -d, -f1`
		INPUTDIR=`grep -w ^${j} ~/Brain_Tumor_2015/Data/${i}/${i}0/dicoms/${i}0_dicomInfo_parsed.csv | cut -d, -f36`
		OUTDIR=/sbia/sbiaprj/brain_tumor/Brain_Tumor_2015/Data/${i}/${i}0/Nifti/${PREFIX}
		
		f=`echo ${PREFIX} | cut -d_ -f1,2`

		#make sure that nifti hasn't been processed already
		if [ ! -e ${OUTDIR}/${PREFIX}.nii.gz ]  
		then

			echo prefix
			echo ${PREFIX}
			echo dicomdir
			echo ${INPUTDIR}
			echo outdir
			echo ${OUTDIR}
	
			if [ ! -d ${OUTDIR} ]
			then
				mkdir -p ${OUTDIR}
			fi
			
#			qsub -l short -j y -o ${OUTDIR}/\$JOB_NAME-\$JOB_ID.log \
			/sbia/sbiaprj/brain_tumor/Brain_Tumor_2015/Scripts/convertDCM.sh \
			${INPUTDIR} \
			${PREFIX} \
			${OUTDIR}

		fi			
	done
done			 
