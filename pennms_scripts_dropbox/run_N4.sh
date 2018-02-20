#!/usr/bin/env bash

for i in `cat /cbica/projects/PsychAnalysis/Lists/Enigma_MASTER_LIST_noDuplicates.csv | grep -v "ID" | grep -v "Exclude" | cut -d, -f1 `
do

	f=`ls -1 /cbica/projects/PsychAnalysis/Data/Enigma/ReOriented/${i}/${i}_LPS.nii.gz`

	if [ -z ${f} ]
	then
		echo "ERROR: Input file not found for ${i}"
		continue
	fi

	outdir=/cbica/projects/PsychAnalysis/Protocols/N4BiasCorrection/Enigma/${i}
	if [ ! -d ${outdir} ]
	then
		mkdir ${outdir}
	fi

	 qsub -l short -j y -o ${outdir}/\$JOB_NAME-\$JOB_ID-${i}.log \
	/cbica/projects/PsychAnalysis/Scripts/N4.sh \
	-in ${f} \
      	-dest ${outdir}

done
