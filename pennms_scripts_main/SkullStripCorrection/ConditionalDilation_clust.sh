#!/bin/sh

mask=$1
input=$2
iter=$3
minsd=$4
maxsd=$5
tolerance=$6

PID=$$
tmp=${SBIA_TMPDIR}/ConditionDilation_${PID}/

export AFNI_NIFTI_NOEXT=YES # AFNI variable for NOT writing out history of afni commands into the header!

CWD=`pwd`
mkdir -p $tmp
cd $tmp

cp -v $mask ${tmp}
cp -v $input ${tmp}

init=`basename ${mask}`

dt=`date +%F-%H%M`
i=1
volRat=100.000
prev=`3dBrickStat -slow -non-zero -volume $init`
current=`3dBrickStat -slow -non-zero -volume $init`
difference=`echo "scale=7; $current - $prev" | bc`

while [ $i -le ${iter} ] && [ `echo "$volRat >= $tolerance" | bc` -eq 1 ]
do
	Stat=`3dBrickStat -slow -non-zero -mean -var -mask $init ${input}`
	mean=`echo $Stat | awk '{ print $1 }'`
	var=`echo $Stat | awk '{ print $2 }'`
	stdev=`echo "scale=7; sqrt( $var )" | bc`

	echo -e "$i \t $mean \t $stdev \t $volRat \t $prev \t $current \t $difference"
	
	3dcalc \
	 -prefix $(basename ${mask%.nii.gz})_dyncond_${minsd}to${maxsd}_${i}.nii.gz \
	 -a $init \
	 -b a+i \
	 -c a-i \
	 -d a+j \
	 -e a-j \
	 -f a+k \
	 -g a-k \
	 -h ${input} \
	 -expr "a+step(amongst(1,a,b,c,d,e,f,g)-a)*and(step((h-${mean})/${stdev}-${minsd}),step(${maxsd}-(h-${mean})/${stdev}))" \
	 -verbose \
	 -nscale \
	 -byte >> tmp_ConditionalDilation_${dt}.log 2>&1;
	suffix=$(basename ${mask%.nii.gz})_dyncond_${minsd}to${maxsd}_${i}.nii.gz
	
	/sbia/home/doshijim/General_Scripts/Morpho.sh \
	 -in $suffix \
	 -open \
	 -kernel 1 >> tmp_ConditionalDilation_${dt}.log 2>&1;
	suffix=${suffix%.nii.gz}_open1mm.nii.gz
	
	thresh=$(( `3dBrickStat -slow -non-zero -count $suffix` / 2 )); 
	3dclust \
	 -prefix ${suffix%.nii.gz}_clust.nii.gz \
	 0 $thresh \
	 $suffix >> tmp_ConditionalDilation_${dt}.log 2>&1;
	suffix=${suffix%.nii.gz}_clust.nii.gz
	
	3dmerge \
	 -1blur_fwhm 2 \
	 -prefix ${suffix%.nii.gz}_s2.nii.gz \
	 $suffix >> tmp_ConditionalDilation_${dt}.log 2>&1;
	suffix=${suffix%.nii.gz}_s2.nii.gz

	3dcalc \
	 -prefix ${suffix%.nii.gz}_thresh.nii.gz \
	 -a $suffix \
	 -expr "step(a-0.5)" \
	 -verbose \
	 -nscale \
	 -byte >> tmp_ConditionalDilation_${dt}.log 2>&1;
	suffix=${suffix%.nii.gz}_thresh.nii.gz 
	
	prev=`3dBrickStat -slow -non-zero -volume $init`
	current=`3dBrickStat -slow -non-zero -volume $suffix`
	volRat=`echo "scale=6; ($current - $prev)/$prev*100" | bc`
	difference=`echo "scale=7; $current - $prev" | bc`
	
	init=$suffix
	(( i++ ))
done

#	/sbia/home/doshijim/General_Scripts/Morpho.sh \
#	 -in $suffix \
#	 -open \
#	 -kernel 1 >> tmp_ConditionalDilation_${dt}.log 2>&1;
#	suffix=${suffix%.nii.gz}_open1mm.nii.gz
#	
#	thresh=$(( `3dBrickStat -slow -non-zero -count $suffix` / 2 )); 
#	3dclust \
#	 -prefix ${suffix%.nii.gz}_clust.nii.gz \
#	 0 $thresh \
#	 $suffix >> tmp_ConditionalDilation_${dt}.log 2>&1;
#	suffix=${suffix%.nii.gz}_clust.nii.gz

mv -v $suffix ${mask%.nii.gz}_CD.nii.gz
rm -fv tmp_ConditionalDilation_${dt}.log
rm -fv $(basename ${mask%.nii.gz})_dyncond_${minsd}to${maxsd}_*.nii.gz

if [ -d $tmp ]
then
	rm -rfv $tmp
fi
