#get sub id
#count tesnor for all 99* subjects visit 1 for dMRI030
basedir=/data/inm1/mapping/RELEASE/1000Brains_derivatives
subject_count=0
tensor_subject_count=0
eddy_subject_count=0
raw_subject_count=0
mask_subject_count=0
visit=1
drctns=30
for i in `ls -d * /data/inm1/mapping/RELEASE/1000Brains_derivatives/99*`
do
	j=`basename $i`
	#echo $j
	#make sure not alphabet
	if [[ $j == 99* ]];then
		((subject_count++))
		if [ -f $basedir/$j/$visit/dwi/tensor/$j_$visit_*MRI0$drctns*Tensor_*_FA* ];then
			((tensor_subject_count++))
			#echo $j
		fi
	        if [ -f $basedir/$j/$visit/dwi/eddy/$j_$visit_*MRI0$drctns*brain_mask* ];then
			((mask_subject_count++))
			#echo $j
		fi
 	
		echo `find $basedir/$j/$visit/dwi/ -maxdepth 20 -name *"MRI0$drctns"*_*brain_mask*`
		
		if [ -f $basedir/$j/$visit/dwi/eddy/"$j"_"$visit"_*MRI0"$drctns"_dwi_eddy.nii.gz ];then
			((eddy_subject_count++))
		fi
		if [ -d "$basedir/$j/1/dwi/raw" ];then
			((raw_subject_count++))
		fi
	fi
done
echo "subjects with raw folder: $raw_subject_count, subjects with tensor fitted: $tensor_subject_count, subjects with eddy: $eddy_subject_count,  subjects with mask: $mask_subject_count, TOTAL SUBJECTS: $subject_count"
		
