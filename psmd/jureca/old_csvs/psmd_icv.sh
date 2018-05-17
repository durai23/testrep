base=/data/inm1/mapping/RELEASE/1000Brains_derivatives
rm -rf ~/psmd_seg_vols.csv
echo "id,gmv,wmv,csfv,icv" >> ~/psmd_seg_vols.csv
psmd_tensor_count=`cat ~/psmd_tensor_list.csv | wc -l`
for i in `cat ~/psmd_tensor_list.csv`
do
	if [ -f "$base"/"$i"/1/CAT/mri/p1"$i"* ] && [ -f "$base"/"$i"/1/CAT/mri/p2"$i"* ] && [ -f "$base"/"$i"/1/CAT/mri/p3"$i"* ]
	then
		gmv=`fslstats "$base"/"$i"/1/CAT/mri/p1"$i"* -V | cut -d" " -f2`
		wmv=`fslstats "$base"/"$i"/1/CAT/mri/p2"$i"* -V | cut -d" " -f2`
		csfv=`fslstats "$base"/"$i"/1/CAT/mri/p3"$i"* -V | cut -d" " -f2`
		#echo $gmv
		#echo $wmv
		#echo $csfv
		icv=`echo $gmv+$wmv+$csfv | bc`
		#echo $icv
		echo "$i,$gmv,$wmv,$csfv,$icv" >> ~/psmd_seg_vols.csv
	else	
		echo "ERROR input file not foudn for atlastr one subject : $i"
		rm -rf ~/psmd_seg_vols.csv
		exit 1
	fi
done
psmd_seg_count=`cat ~/psmd_seg_vols.csv | wc -l`
if [ $psmd_tensor_count -eq $psmd_seg_count ]
then
	echo "QC OK"
else
	echo "QC FAIL"
fi
