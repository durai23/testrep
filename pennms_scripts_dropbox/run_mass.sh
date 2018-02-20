#!/usr/bin/env bash

for i in `cat /cbica/projects/PsychAnalysis/Lists/Enigma_MASTER_LIST_noDuplicates.csv | grep -v "ID" | grep -v "Exclude" | cut -d, -f1 `
do

	f=`ls -1 /cbica/projects/PsychAnalysis/Protocols/N4BiasCorrection/Enigma/${i}/${i}_LPS_N4.nii.gz `

	if [ -z ${f} ]
	then
		echo "ERROR: Input file not found for ${i}"
		continue
	fi

	outdir=/cbica/projects/PsychAnalysis/Protocols/SkullStripping/Enigma/${i}
	if [ ! -d ${outdir} ]
	then
		mkdir ${outdir}
	fi

	# qsub -j y -o ${outdir}/\$JOB_NAME-\$JOB_ID-${i}.log \
	/cbica/software/lab/mass/1.1.0/bin/mass \
	-in ${f} \
	-ref /cbica/projects/PsychAnalysis/Protocols/SkullStripping/Enigma/template_folder \
	-regs 14 \
      	-dest ${outdir}

done
