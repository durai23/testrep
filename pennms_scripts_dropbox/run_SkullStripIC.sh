#!/usr/bin/env bash


for i in `cat /cbica/projects/PsychAnalysis/Lists/Enigma_MASTER_LIST_noDuplicates.csv | grep -v "ID" | grep -v "Exclude" | cut -d, -f1 `
do

	JACRNK=`command ls -1 ${PWD}/${i}/${i}_LPS_N4_brain_JacRank.nii.gz`
	INPUT=`command ls -1 /cbica/projects/PsychAnalysis/Protocols/N4BiasCorrection/Enigma/${i}/${i}_LPS_N4.nii.gz `

	if [ -z ${INPUT} ] || [ -z ${JACRNK} ] 
	then
		echo "ERROR: File not found for  ${i} "
		continue
	fi
	
	OUTDIR=${PWD}/${i}

	qsub -j y -o ${OUTDIR}/\$JOB_NAME-\$JOB_ID_${i}.log \
	/cbica/projects/PsychAnalysis/Scripts/SkullStripCorrection/SkullStrip_IC.sh \
	-input ${INPUT} \
	-JacRank ${JACRNK} \
	-scripts /cbica/projects/PsychAnalysis/Scripts/SkullStripCorrection

done
