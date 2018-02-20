#!/bin/sh

module unload python
module load python/2.5.2


input=$1
output=$2

dest=`dirname $output`

if [ ! -f ${input} ]
then
	echo -e "\nInput file doesn't exist!"
	exit 1
fi

if [ ! -f ${output} ]
then
	mkdir -pv ${dest}

	nifti1_test \
	 -n2 ${input} \
	 ${output%.nii.gz}_pair;
	
	nifti_tool \
	 -mod_hdr \
	 -mod_field sform_code 0 \
	 -prefix ${output%.nii.gz}_pair_nosform.hdr \
	 -infiles ${output%.nii.gz}_pair.hdr;
	
	3dresample \
	 -orient rai \
	 -prefix ${output%.nii.gz}_pair_nosform_LPS.hdr \
	 -inset ${output%.nii.gz}_pair_nosform.hdr;
	 
	cp -v \
	 ${output%.nii.gz}_pair_nosform_LPS.img \
	 ${output%.nii.gz}_pair_LPS.img;
	 
	makeNiftiHeader.py \
	 -d ${output%.nii.gz}_pair_LPS.img \
	 -c ${output%.nii.gz}_pair_nosform_LPS.hdr \
	 -r LPS \
	 -o ${dest}/ \
	 -v;

	nifti_tool \
	 -mod_hdr \
	 -mod_field sform_code 0 \
	 -prefix ${output%.nii.gz}_pair_LPS_nosform.hdr \
	 -infiles ${output%.nii.gz}_pair_LPS.hdr;
	
	rm -fv ${output%.nii.gz}_pair.hdr ${output%.nii.gz}_pair.img 
	rm -fv ${output%.nii.gz}_pair_nosform.hdr ${output%.nii.gz}_pair_nosform.img 
	rm -fv ${output%.nii.gz}_pair_nosform_LPS.hdr ${output%.nii.gz}_pair_nosform_LPS.img
	rm -fv ${output%.nii.gz}_pair_LPS.hdr ${output%.nii.gz}_pair_LPS.img
	
	nifti1_test \
	 -zn1 \
	 ${output%.nii.gz}_pair_LPS_nosform.img \
	 ${output%.nii.gz};
	 
	rm -fv ${output%.nii.gz}_pair_LPS_nosform.img ${output%.nii.gz}_pair_LPS_nosform.hdr;
else
	echo -e "\nOutput file already exists!"
	exit 1
fi 

exit 0

