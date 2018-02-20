#!/bin/sh -x

. /sbia/home/doshijim/General_Scripts/BashUtilityFunctions.sh


################################################ FUNCTIONS ################################################

help()
{
cat <<HELP

This script does the following:



##############################################
USAGE :	$0 [OPTIONS]
OPTIONS:

Reqd:	-JacRank 	   < file >	absolute path to the input JacobianRankMask
	-input	 	   < file >	absolute path to the input T1 image 

Opt:	-erode	 	   < int >	erosion kernel size (default: 2)
	-iter	 	   < int >	number of iterations of conditional dilation (default: 100)
	-minsd	 	   < float >	minimum std dev to accept during conditional dilation (default: -1)
	-maxsd	 	   < float >	maximum std dev to accept during conditional dilation (default: 3)
	-tolerance 	   < float >	minimum tolerance level (default: 0.01)
	-csf	 	   < float >	csf weight to use for fuzzy segmentation (default: 1.0)
	-absthresh	   < float >	threshold value for creating the final binary mask (default: 2.25 {0..3})
	-scripts	   < path >	absolute path to the scripts folder (default: `dirname $0`)

ERROR: Not enough arguments!!
##############################################

HELP
exit 1
}

cleanUpandExit()
{
	echo -e ":o:o:o:o:o Aborting Operations .... \n\n"
	
	if [ -d "$TMP" ]
	then
		BGjobs=`jobs -p`
		if [ -n "$BGjobs" ]
		then
			kill -s SIGINT $BGjobs
		fi

		if [ "$verbose" == "1" ]
		then
			rm -rfv ${TMP}
		else
			rm -rf ${TMP}
		fi
	fi
	
	executionTime		
	exit 1
}

parse()
{
	while [ -n "$1" ]; do
		case $1 in
			-h) 
				help;
				shift 1;;			# help is called
		     	-JacRank) 
				JacRank=$2;
				checkFile $JacRank; 

				temp=`FileAtt $JacRank`				
				JacRankExt=`echo $temp | awk '{ print $1 }'`
				JacRankbName=`echo $temp | awk '{ print $2 }'`

				shift 2;;			# SubID is set
		     	-input) 
				input=$2;
				checkFile $input; 

				temp=`FileAtt $input`				
				inputExt=`echo $temp | awk '{ print $1 }'`
				inputbName=`echo $temp | awk '{ print $2 }'`

				shift 2;;			# SubID is set
			-erode) 
				erode=$2;
				shift 2;;			# source path is set
			-iter) 
				iter=$2;
				shift 2;;			# source path is set
			-minsd) 
				minsd=$2;
				shift 2;;			# source path is set
			-maxsd) 
				maxsd=$2;
				shift 2;;			# source path is set
			-tolerance) 
				tolerance=$2;
				shift 2;;			# source path is set
			-csf) 
				csf=$2;
				shift 2;;			# source path is set
		     	-scripts) 
				scripts=`checkPath $2`;
				shift 2;;			# SubID is set
			-absthresh) 
				absthresh=$2;
				shift 2;;			# source path is set
			-V) 
				print_version $0 --copyright '2012, 2013 University of Pennsylvania'; 
				exit 0;;
			-*) 
				echo "ERROR: no such option $1";
				help;;
			 *) 
				break;;
		esac
	done
}

import()
{
	ext=$1
	inFile=$2
	outFile=$3
	
	if [ ! -f ${outFile}.nii.gz ]
	then
		if [ "${ext}" == "nii.gz" ] || [ "${ext}" == "nii" ] || [ "${ext}" == "img" ]
		then
			echoV "nifti1_test -zn1 ${inFile} ${outFile}"
			nifti1_test -zn1 ${inFile} ${outFile}
		elif [ "${ext}" == "hdr" ]
		then
			echoV "nifti1_test -zn1 ${inFile%.hdr}.img ${outFile}"
			nifti1_test -zn1 ${inFile%.hdr}.img ${outFile}
		fi	
	fi
}

################################################ END OF FUNCTIONS ################################################

################################################ MAIN BODY ################################################

if [ $# -lt 1 ]; then
	help
fi

### Timestamps
startTime=`date +%F-%H:%M:%S`
startTimeStamp=`date +%s`

echo -e "\nRunning commands on		: `hostname`"
echo -e "Start time			: ${startTime}\n"

### Default Parameters
erode=2
csf=1.0
absthresh=2.25
perc=99
iter=100
minsd=-1
maxsd=3
tolerance=0.01
scripts=`dirname $0`/
verbose=0


### Specifying the trap signal
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGHUP signal'" SIGHUP 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGINT signal'" SIGINT 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGTERM signal'" SIGTERM 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGKILL signal'" SIGKILL

### Reading the arguments
echo -e "Parsing arguments		: $*"
parse $*

### Sanity check on parameters
if [ -z "$dest" ]
then
	dest=`dirname $JacRank`/
fi

### Scripts and environment variables
FSLOUTPUTTYPE=NIFTI_GZ; export $FSLOUTPUTTYPE

### Forming FileNames
# TMP
PID=$$

createTempDir SkullStripIC $PID
echoV "\n----->	Temporary local directory created at $TMP ...\n" 1

### Importing required input files to the temp dir
echoV "----->	Importing the required input files ..." 1
import ${JacRankExt} ${JacRank} ${TMP}${JacRankbName}
import ${inputExt} ${input} ${TMP}${inputbName}

cd $TMP

### Bias Correct input
echoV "----->	Running N3 bias correction ..." 1
${scripts}N3.sh \
 -in ${TMP}${inputbName}.nii.gz \
 -dest ${TMP} >> ${TMP}Debug.log 2>&1

### Calculate 100% mask
echoV "----->	Generating a 99% agreement mask ..." 1
p100=`3dBrickStat -slow -max ${TMP}${JacRankbName}.nii.gz`
pCut=`echo "scale=2; $p100 * ${perc} / 100" | bc`
3dcalc \
 -prefix ${TMP}ThreshMask.nii.gz \
 -a ${TMP}${JacRankbName}.nii.gz \
 -expr "step(a-${pCut})" \
 -verbose \
 -nscale \
 -byte >> ${TMP}Debug.log 2>&1

### Threshold the input image at threshold mask and segment it and then add gm and wm labels
echoV "----->	Thresholding the input image at the 99% agreement mask and segmenting it ..." 1
3dcalc \
 -prefix ${TMP}${inputbName}_n3_thresh.nii.gz \
 -a ${TMP}${inputbName}_n3.nii.gz \
 -b ${TMP}ThreshMask.nii.gz \
 -expr 'step(b)*a' \
 -verbose \
 -nscale >> ${TMP}Debug.log 2>&1
 
mico \
 -v \
 -c ${csf} \
 -o ${TMP} \
 --zn1 \
 ${TMP}${inputbName}_n3_thresh.nii.gz >> ${TMP}Debug.log 2>&1

### Threshold the input image at 100% mask and segment it
echoV "----->	Thresholding the input image at the 1% agreement mask and segmenting it ..." 1
3dcalc \
 -prefix ${TMP}${inputbName}_n3_Fullthresh.nii.gz \
 -a ${TMP}${inputbName}_n3.nii.gz \
 -b ${TMP}${JacRankbName}.nii.gz \
 -expr 'step(b)*a' \
 -verbose \
 -nscale >> ${TMP}Debug.log 2>&1
 
mico \
 -v \
 -c ${csf} \
 -o ${TMP} \
 --fuzzy \
 --zn1 \
 ${TMP}${inputbName}_n3_Fullthresh.nii.gz >> ${TMP}Debug.log 2>&1

### Combining the GM and WM masks
echoV "----->	Adding GM and WM masks ..." 1
3dcalc \
 -prefix ${TMP}${inputbName}_n3_thresh_labels_gm+wm.nii.gz \
 -a ${TMP}${inputbName}_n3_thresh_labels.nii.gz"<150>" \
 -b ${TMP}${inputbName}_n3_thresh_labels.nii.gz"<250>" \
 -expr 'step(a+b)' \
 -verbose \
 -nscale \
 -byte >> ${TMP}Debug.log 2>&1

3dcalc \
 -prefix ${TMP}${inputbName}_n3_Fullthresh_gm+wm.nii.gz \
 -a ${TMP}${inputbName}_n3_Fullthresh_gm.nii.gz \
 -b ${TMP}${inputbName}_n3_Fullthresh_wm.nii.gz \
 -expr 'a+b' \
 -verbose \
 -nscale \
 -byte >> ${TMP}Debug.log 2>&1

### Calculate mean and stdev within brain and non-brain regions
echoV "----->	Gathering statistics from input intensity image ..." 1
Stats=`3dBrickStat -slow -mean -var -mask ${TMP}${inputbName}_n3_thresh_labels_gm+wm.nii.gz ${TMP}${inputbName}_n3.nii.gz`
BrainMean=`echo $Stats | awk '{ print $1 }'`
BrainVar=`echo $Stats | awk '{ print $2 }'`
BrainStdev=`echo "scale=4; sqrt($BrainVar)" | bc`
#echo $BrainMean $BrainStdev

Stats=`3dBrickStat -slow -mean -var -mask ${TMP}${inputbName}_n3_thresh_labels.nii.gz"<10>" ${TMP}${inputbName}_n3.nii.gz`
NonBrainMean=`echo $Stats | awk '{ print $1 }'`
NonBrainVar=`echo $Stats | awk '{ print $2 }'`
NonBrainStdev=`echo "scale=4; sqrt($NonBrainVar)" | bc`
#echo $NonBrainMean $NonBrainStdev

### Calculate absolute z-scores
echoV "----->	Calculating absolute z-scores ..." 1
3dcalc \
 -prefix ${TMP}Zscores_Brain.nii.gz \
 -a ${TMP}${inputbName}_n3.nii.gz \
 -b ${TMP}${JacRankbName}.nii.gz \
 -expr "step(b)*abs((a-${BrainMean})/${BrainStdev})" \
 -verbose \
 -nscale \
 -float >> ${TMP}Debug.log 2>&1
 
3dcalc \
 -prefix ${TMP}Zscores_NonBrain.nii.gz \
 -a ${TMP}${inputbName}_n3.nii.gz \
 -b ${TMP}${JacRankbName}.nii.gz \
 -expr "step(b)*abs((a-${NonBrainMean})/${NonBrainStdev})" \
 -verbose \
 -nscale \
 -float >> ${TMP}Debug.log 2>&1

### Invert z-scores
echoV "----->	Inverting the absolute z-scores ..." 1
BrainMax=`3dBrickStat -slow -non-zero -max ${TMP}Zscores_Brain.nii.gz`
NonBrainMax=`3dBrickStat -slow -non-zero -max ${TMP}Zscores_NonBrain.nii.gz`
e2=`echo "scale=4; e(2)" | bc -l`

3dcalc \
 -prefix ${TMP}Zscores_Brain_inv_exp2.nii.gz  \
 -a ${TMP}Zscores_Brain.nii.gz \
 -expr "step(a)*exp(2*step(a)*(1-a/${BrainMax}))/${e2}" \
 -verbose \
 -nscale \
 -float >> ${TMP}Debug.log 2>&1
 
3dcalc \
 -prefix ${TMP}Zscores_NonBrain_inv_exp2.nii.gz \
 -a ${TMP}Zscores_NonBrain.nii.gz \
 -expr "step(a)*exp(2*step(a)*(1-a/${NonBrainMax}))/${e2}" \
 -verbose \
 -nscale \
 -float >> ${TMP}Debug.log 2>&1

### Divide the brain z-scores to non-brain z-scores
echoV "----->	Suppressing the CSF z-scores ..." 1
3dcalc \
 -prefix ${TMP}Zscores_Brain_inv_exp2_divideNonBrain.nii.gz \
 -a ${TMP}Zscores_Brain_inv_exp2.nii.gz \
 -b ${TMP}Zscores_NonBrain_inv_exp2.nii.gz \
 -expr 'a/b' \
 -verbose \
 -nscale \
 -float >> ${TMP}Debug.log 2>&1

### Adding the JacRank and the Exp2AbsInvZscore map
echoV "----->	Combining the JacobianRankMask, Fuzzy segmentation and Intensity Correction ..." 1
JacMax=`3dBrickStat -slow -max ${TMP}${JacRankbName}.nii.gz`
ZMax=`3dBrickStat -slow -max ${TMP}Zscores_Brain_inv_exp2_divideNonBrain.nii.gz`
FuzzMax=`3dBrickStat -slow -max ${TMP}${inputbName}_n3_Fullthresh_gm+wm.nii.gz`
FuzzCSFMax=`3dBrickStat -slow -max ${TMP}${inputbName}_n3_Fullthresh_csf.nii.gz`

3dcalc \
 -prefix ${JacRank%.nii.gz}_IC+Fuzzy_C${csf}.nii.gz \
 -a ${TMP}${JacRankbName}.nii.gz \
 -b ${TMP}Zscores_Brain_inv_exp2_divideNonBrain.nii.gz \
 -c ${TMP}${inputbName}_n3_Fullthresh_gm+wm.nii.gz \
 -d ${TMP}${inputbName}_n3_Fullthresh_csf.nii.gz \
 -e ThreshMask.nii.gz \
 -expr "a/${JacMax}+b/${ZMax}+c/${FuzzMax}+step(e)*(d/${FuzzCSFMax})" \
 -verbose \
 -nscale \
 -float >> ${TMP}Debug.log 2>&1

### Thresholding the JacobianRankMask
echoV "----->	Thresholding the final brain weight map at $absthresh ..." 1
${scripts}ThresholdJacobian_new.sh \
 -in ${input} \
 -dest ${TMP} \
 -jacRank ${JacRank%.nii.gz}_IC+Fuzzy_C${csf}.nii.gz \
 -absThresh $absthresh \
 -mask ${inputbName}_cbq_mask_${absthresh} \
 -cbq ${inputbName}_str_cbq_${absthresh} \
 -kernel ${erode} \
 -exe ${scripts} >> ${TMP}Debug.log 2>&1

### Cluster and keep only connected components/clusters
echoV "----->	Clustering the thresholded mask before dilating ..." 1
thresh=$(( `3dBrickStat -slow -non-zero -count ${TMP}${inputbName}_cbq_mask_${absthresh}.nii.gz` / 2 )); 
3dclust \
 -prefix ${TMP}${inputbName}_cbq_mask_${absthresh}_clust.nii.gz \
 0 $thresh \
 ${TMP}${inputbName}_cbq_mask_${absthresh}.nii.gz >> ${TMP}Debug.log 2>&1

### Conditional Dilation
echoV "----->	Running a condition dilation on the thresholded mask ..." 1
${scripts}ConditionalDilation_clust.sh \
 ${TMP}${inputbName}_cbq_mask_${absthresh}_clust.nii.gz \
 ${JacRank%.nii.gz}_IC+Fuzzy_C${csf}.nii.gz \
 $iter \
 $minsd $maxsd \
 $tolerance >> ${TMP}Debug.log 2>&1
 
before=`3dBrickStat -slow -non-zero -volume ${TMP}${inputbName}_cbq_mask_${absthresh}_clust.nii.gz`
after=`3dBrickStat -slow -non-zero -volume ${TMP}${inputbName}_cbq_mask_${absthresh}_clust_CD.nii.gz`
ratio=`echo "scale=4; ($after - $before) / $before * 100" | bc`
echoV "\t-->	Percent change after CD: $ratio % ..." 1

### Additional Conditional Dilation based on GM and CSF intensities
echoV "----->	Running additional dilation for GM and CSF ..." 1

suffix=${TMP}${inputbName}_cbq_mask_${absthresh}_clust_CD.nii.gz

for t in 150 10
do
	for tt in {1..2}
	do
		mean=`3dBrickStat -slow -non-zero -mean -mask ${TMP}${inputbName}_n3_thresh_labels.nii.gz"<${t}>" ${TMP}${inputbName}_n3.nii.gz`

		3dcalc \
		 -prefix ${suffix%.nii.gz}_dil${t}.nii.gz \
		 -a $suffix \
		 -b a+i \
		 -c a-i \
		 -d a+j \
		 -e a-j \
		 -f a+k \
		 -g a-k \
		 -h ${TMP}${inputbName}_n3.nii.gz \
		 -expr "a+step(amongst(1,a,b,c,d,e,f,g)-a)*and(step(h-0.6*${mean}),step(1.6*${mean}-h))" \
		 -verbose \
		 -nscale \
		 -byte >> ${TMP}Debug.log 2>&1

		thresh=$(( `3dBrickStat -slow -non-zero -count ${suffix%.nii.gz}_dil${t}.nii.gz` / 2 )) 
		3dclust -prefix ${suffix%.nii.gz}_dil${t}_clust.nii.gz 0 $thresh ${suffix%.nii.gz}_dil${t}.nii.gz  >> ${TMP}Debug.log 2>&1

		3dmerge \
		 -1blur_fwhm 2 \
		 -prefix ${suffix%.nii.gz}_dil${t}_clust_s2.nii.gz \
		 ${suffix%.nii.gz}_dil${t}_clust.nii.gz >> ${TMP}Debug.log 2>&1

		3dcalc \
		 -prefix ${suffix%.nii.gz}_dil${t}_clust_s2_thresh.nii.gz \
		 -a ${suffix%.nii.gz}_dil${t}_clust_s2.nii.gz \
		 -expr "step(a-0.5)" \
		 -verbose \
		 -nscale \
		 -byte >> ${TMP}Debug.log 2>&1
		 
		${scripts}FillHoles_3D.sh -in ${suffix%.nii.gz}_dil${t}_clust_s2_thresh.nii.gz -dest ${TMP}  >> ${TMP}Debug.log 2>&1
		 
		suffix=${suffix%.nii.gz}_dil${t}_clust_s2_thresh_filled.nii.gz
	done
done

### Moving the results to the destination directory
echoV "----->	Moving the results to the destination directory ..." 1
if [ ! -d ${dest} ]
then
	if [ "$verbose" == "1" ]	
	then
		mkdir -pv $dest
	else
		mkdir -p $dest
	fi
fi

mvV $suffix ${dest}/${inputbName}_str_mask_CD_C${csf}_t${absthresh}Dyn${minsd}to${maxsd}.nii.gz

3dcalc \
 -prefix ${dest}/${inputbName}_str_CD_C${csf}_t${absthresh}Dyn${minsd}to${maxsd}.nii.gz \
 -a $input \
 -b ${dest}/${inputbName}_str_mask_CD_C${csf}_t${absthresh}Dyn${minsd}to${maxsd}.nii.gz \
 -expr "a*step(b)" \
 -verbose \
 -nscale >> ${TMP}Debug.log 2>&1

### Removing the temporary directory
if [ -d $TMP ]
then
	rm -rf $TMP
fi

### Execution Time 
executionTime $startTimeStamp

################################################ END ################################################
