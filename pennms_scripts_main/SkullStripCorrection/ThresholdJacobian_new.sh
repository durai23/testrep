#!/bin/bash

################################################ VERSION INFO ################################################
# $Id: ThresholdJacobian.sh 71 2011-11-02 17:16:08Z doshijim@UPHS.PENNHEALTH.PRV $
#
version()
{
	# Display the version number and date of modification extracted from
	# the Id variable.
	SVNversion="$Id: ThresholdJacobian.sh 71 2011-11-02 17:16:08Z doshijim@UPHS.PENNHEALTH.PRV $"
	Auth="Jimit Doshi"
	ver="$Rev: 71 $"
	mod="$LastChangedDate: 2011-11-02 13:16:08 -0400 (Wed, 02 Nov 2011) $"
	echo -e "Author			: $Auth"
	echo -e "Revision		$ver"
	echo -e "Last Modification	$mod"
#	echo -e "$0 version \c"
#	echo $SVNversion|cut -f3,4,5 -d" "
	exit 5
}

################################################ FUNCTIONS ################################################

help()
{
cat <<HELP

This script does the following:



##############################################
USAGE :	$0 [OPTIONS]
OPTIONS:

Reqd:	-in	   < file >	absolute path of the input file to be skull-stripped and cerebellum removed
	-jacRank   < file >	absolute path of the Jacobian Ranked mask

Opt:	-dest	   < path >	absolute path to the destination where the results are to be stored (default: same as input)
	-tmp	   < path >	absolute path to the temporary directory (default: \$SBIA_TMPDIR )
	-perThresh < float >	Percent Threshold for the aggresiveness of the skull-stripping and cerebellum removal. 0 < \$perThresh < 100 (default: 50)
	-absThresh < float >	Absolute Threshold for the aggresiveness of the skull-stripping and cerebellum removal. 0 < \$absThresh < max (no default)
				If this argument is provided, it will override the -perThresh value.
	-mask      < pattern >	Prefix of the output brain mask (default: input_cbq_mask)
				Prefix of the output ventricle mask, if "-vnmask" is set (default: input_cbq_vnmask)
				Provide the full filename without the extension or the path
	-cbq	   < pattern >	Prefix of the output skull-stripped, cerebellum removed image (default: input_str_cbq)
	-kernel    < int >	Spherical dilation kernel size, in mm (default: 6mm)
	-exe	   < path >	absolute path to the directory containing the scripts (default: `dirname $0`)
	-vnmask	   < 0/1 >	flag to signify ventricle mask generation. If set, "-in" and "-cbq" options become invalid. (default: 0 - no VNmask)
	-v	   		verbose output (default: no output)
	-V			Version info


ERROR: Not enough arguments!!
##############################################

DEPENDENCIES:
	3dcalc		: `which 3dcalc`
	3dBrickStat	: `which 3dBrickStat`
	nifti1_test	: `which nifti1_test`
	3dclust		: `which 3dclust`
	
	fslmaths	: `which fslmaths`

HELP
exit 1
}

checkDependency()
{
	pth=`which $1 2>&1`
	if [ $? != 0 ]
	then
		echo -e "${1} not installed OR not found. Aborting operations ..."
		cleanUpandExit
	fi
}

checkExitCode()
{
	if [ $1 != 0 ]
	then
		echo -e $2
		cleanUpandExit
	fi
}

cleanUpandExit()
{
	echo -e ":o:o:o:o:o Aborting Operations .... \n\n"
	
	if [ -d "$TMP" ]
	then
		if [ "$TMP" != "$dest" ]
		then
			rm -rfv ${TMP}
		else
			rmV ${Thresholded}
			rmV ${Sum_open}
			rmV ${Filled}
			rmV ${Clustered}

			if [ "$CpInput" == "1" ]
			then
				rmV ${InbName}.nii.gz
			fi
	
			if [ "$CpJacob" == "1" ]
			then
				rmV ${JRbName}.nii.gz
			fi
		fi
	fi
	
	executionTime
	
	exit 1
}

checkPath()
{
	path=`echo ${1##*/}`
	
	if [ -n "$path" ]
	then
		echo ${1}/
	else
		echo $1
	fi
}

checkFile()
{
	if [ ! -f $1 ]
	then
		echo -e "\nERROR: Input file $1 does not exist! Aborting operations ..."
		cleanUpandExit
	fi
}

FileAtt()
{
	IP=$1;
	
	if [ ! -f ${IP} ]
	then
		echo -e "\nERROR: Input file $IP does not exist!"
		cleanUpandExit
	fi

	ext=`echo ${IP##*.}`
	bName=`basename ${IP%.${ext}}`
	
	if [ "$ext" == "gz" ]
	then
		ext=`echo ${bName##*.}`.${ext}
		bName=`basename ${IP%.${ext}}`
	fi
	
	if [ "$ext" != "nii.gz" ] && [ "$ext" != "hdr" ] && [ "$ext" != "img" ] && [ "$ext" != "nii" ]
	then
		echo -e "\nERROR: Input file extension $ext not recognized! Please check ..."
		cleanUpandExit
	fi
	
	echo $ext $bName
}

executionTime()
{
	endTimeStamp=`date +%s`
	total=$[ (${endTimeStamp} - ${startTimeStamp})]
	if [ ${total} -gt 60 ]
	then
		if [ ${total} -gt 3600 ]
		then
			if [ ${total} -gt 86400 ]
			then
				echoV "\nExecution time:  $[ ${total} / 86400]d $[ ${total} % 86400 / 3600]h $[ ${total} % 86400 % 3600 / 60]m $[ ${total} % 86400 % 3600 % 60]s"
			else
				echoV "\nExecution time:  $[ ${total} / 3600]h $[ ${total} % 3600 / 60]m $[ ${total} % 3600 % 60]s"
			fi
		else
			echoV "\nExecution time:  $[ ${total} / 60]m $[ ${total} % 60]s"
		fi
	else
		echoV "\nExecution time:  $[ ${total} % 60]s"
	fi
}

parse()
{
	while [ -n "$1" ]; do
		case $1 in
			-h) 
				help;
				shift 1;;			# help is called
		     	-in) 
				input=$2;
				
				if [ ! -f $input ]
				then
					echo -e "\nERROR: Input file $input does not exist! Aborting operations ..."
					exit 1
				fi
				
				temp=`FileAtt $input`				
				InExt=`echo $temp | awk '{ print $1 }'`
				InbName=`echo $temp | awk '{ print $2 }'`

				shift 2;;			# SubID is set
			-dest) 
				dest=`checkPath $2`;
				shift 2;;			# source path is set
			-tmp) 
				tmp=`checkPath $2`;
				shift 2;;			# source path is set
			-perThresh) 
				perThresh=$2;
				shift 2;;			# source path is set
			-absThresh) 
				absThresh=$2;
				shift 2;;			# source path is set
			-kernel) 
				kernel=$2;
				shift 2;;			# source path is set
			-mask) 
				mask=$2;
				shift 2;;			# source path is set
			-cbq) 
				cbq=$2;
				shift 2;;			# source path is set
			-exe) 
				scripts=`checkPath $2`;
				shift 2;;			# source path is set
			-jacRank) 
				jacRank=$2;
				
				if [ ! -f $jacRank ]
				then
					echo -e "\nERROR: Input file $jacRank does not exist! Aborting operations ..."
					exit 1
				fi

				temp=`FileAtt $jacRank`				
				JRExt=`echo $temp | awk '{ print $1 }'`
				JRbName=`echo $temp | awk '{ print $2 }'`

				shift 2;;			# source path is set
			-vnmask) 
				vnmask=$2;
				shift 2;;			# source path is set
			-v) 
				verbose=1;
				shift 1;;			# source path is set
			-V) 
				version
				shift 1;;			# source path is set
			-*) 
				echo "ERROR: no such option $1";
				help;;
			 *) 
				break;;
		esac
	done
}

convertToNifti()
{
	in=$1
	if [ -z "$2" ]
	then
		out=$1
	else
		out=$2
	fi
	
	
	nifti1_test -zn1 $in $out

	if [ -f ${out%.img}.nii.gz ]
	then
		echoV "\nConverted to NIFTIGZ: $out"
		rm -fv ${in} ${in%.img}.hdr
	else
		echoV "\nConversion to NIFTIGZ failed: $in"
	fi
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
			echoV "nifti1_test -zn1 ${inFile} ${outFile}" 1>&2
			nifti1_test -zn1 ${inFile} ${outFile}
			echo 1		### Returning a value indicating that the file was copied successfully
		elif [ "${ext}" == "hdr" ]
		then
			echoV "nifti1_test -zn1 ${inFile%.hdr}.img ${outFile}" 1>&2
			nifti1_test -zn1 ${inFile%.hdr}.img ${outFile}
			echo 1		### Returning a value indicating that the file was copied successfully
		fi	
	fi
}

echoV()
{
	if [ "$verbose" == "1" ]
	then
		echo -e $1
	fi
}

rmV()
{
	if [ -f $1 ]
	then
		if [ "$verbose" == "1" ]
		then
			rm -fv $1
		else
			rm -f $1
		fi
	fi
}
################################################ END OF FUNCTIONS ################################################

################################################ MAIN BODY ################################################

if [ $# -lt 4 ]; then
	help
fi

### Timestamps
startTime=`date +%F-%H:%M:%S`
startTimeStamp=`date +%s`

echo -e "\nRunning commands on	: `hostname`"
echo -e "Start time		: ${startTime}\n"

### Default Parameters
perThresh=50
absThresh=''
kernel=6
verbose=1
vnmask=0
scripts=`dirname $0`/
FSLOUTPUTTYPE=NIFTI_GZ; export $FSLOUTPUTTYPE

### Specifying the trap signal
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGHUP signal'" SIGHUP 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGINT signal'" SIGINT 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGTERM signal'" SIGTERM 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGKILL signal'" SIGKILL

### Reading the arguments
echo -e "Parsing arguments	: $*"
parse $*

### Checking for default parameters
if [ -z $dest ]
then
	dest=`dirname $input`/
fi

if [ -z $mask ]
then
	if [ "$vnmask" == "0" ]
	then
		mask=${InbName}_cbq_mask
	else
		mask=${InbName}_cbq_vnmask
	fi
fi

if [ -z $cbq ]
then
	cbq=${InbName}_str_cbq
fi

fillholes=${scripts}FillHoles_3D.sh
Morpho=${scripts}Morpho.sh

### Check if all dependenices are satisfied
checkDependency	3dcalc
checkDependency	nifti1_test
checkDependency	3dBrickStat
checkDependency	3dclust
		
checkDependency	fslmaths

### Forming FileNames
# TMP
PID=$$

if [ -n "$tmp" ]
then
	if [ ! -d "$tmp" ]
	then
		mkdir -p $tmp
	fi

#	TMP=`mktemp -d -p ${tmp} ThresholdMask_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; exit 1; }
	TMP=$tmp
elif [ -n "$SBIA_TMPDIR" ]
then
	if [ ! -d "$SBIA_TMPDIR" ]
	then
		mkdir -p $SBIA_TMPDIR
	fi

	TMP=`mktemp -d -p ${SBIA_TMPDIR} ThresholdMask_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; exit 1; }
else
	TMP=`mktemp -d -t ThresholdMask_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; exit 1; }
fi
echoV "\n----->	Temporary local directory created at $TMP ...\n"

# Output Images
Thresholded=${JRbName}_thresh.nii.gz
Sum_ero=${Thresholded%.nii.gz}_ero${kernel}mm.nii.gz
Sum_ero_clust=${Sum_ero%.nii.gz}_clust.nii.gz
Sum_ero_clust_dil=${Sum_ero_clust%.nii.gz}_dil${kernel}mm.nii.gz
Sum_open=${Sum_ero_clust_dil}

Filled=${Sum_open%.nii.gz}_filled.nii.gz
Clustered=${Filled%.nii.gz}_clustered.nii.gz


if [ "$verbose" == "1" ]
then
	echo -e "\nINPUT FILES"
	if [ "$vnmask" == "0" ]
	then
		echo -e "Input Image		: ${input}"
	fi
	echo -e "Jacobian Ranked Mask	: ${jacRank}"

	echo -e "\nOUTPUT FILES"
	if [ "$vnmask" == "0" ]
	then
		echo -e "Final Brain Mask	: ${dest}${mask}.nii.gz"
		echo -e "Final CBQ image		: ${dest}${cbq}.nii.gz"
	else
		echo -e "Final Ventricle Mask	: ${dest}${mask}.nii.gz"
	fi

	echo -e "\nPARAMETERS"
	
	if [ -n "$absThresh" ]
	then
		echo -e "Absolute Threshold	: $absThresh"
	else
		echo -e "Percent Threshold	: $perThresh %"
	fi
	echo -e "Dilation Kernel Size	: ${kernel}mm"
fi

### Importing data to the temporary directory
echoV "\n----->	Importing required files to the temporary local directory ...\n"
CpInput=0
CpJacob=0

if [ "$vnmask" == "0" ]
then
	CpInput=`import ${InExt} ${input} ${TMP}${InbName}`
fi
CpJacob=`import ${JRExt} ${jacRank} ${TMP}${JRbName}`

cd $TMP

### Thresholding the Jacobian ranked reference masks
if [ -n "$absThresh" ]
then
	thresh=$absThresh
else
	max=`3dBrickStat -slow -max ${JRbName}.nii.gz`
	thresh=`echo "scale=7; $perThresh / 100 * ${max}" | bc`
fi

echoV "\n----->	Thresholding the Jacobian ranked reference mask at ${thresh} ...\n"

echoV "\n3dcalc \n
 -prefix ${Thresholded} \n
 -a ${JRbName}.nii.gz \n
 -expr 'step(a-$thresh)' \n
 -nscale \n
 -byte \n
 -verbose;"

if [ "$verbose" == "1" ]
then
	3dcalc \
	 -prefix ${Thresholded} \
	 -a ${JRbName}.nii.gz \
	 -expr "step(a-$thresh)" \
	 -nscale \
	 -byte \
	 -verbose;
else
	3dcalc \
	 -prefix ${Thresholded} \
	 -a ${JRbName}.nii.gz \
	 -expr "step(a-$thresh)" \
	 -nscale \
	 -byte;
fi
checkExitCode $? "\nERROR: Thresholding of the Jacobian Rank Mask failed!!!"

### Opening the thresholded mask
if [ "$kernel" != 0 ]
then
	echoV "\n----->	Eroding the thresholded mask ...\n"
	echoV "${Morpho} \n
	 -in ${Thresholded} \n
	 -erode \n
	 -dest $TMP \n
	 -kernel $kernel \n
	 -v;"

	if [ "$verbose" == "1" ]
	then
		${Morpho} \
		 -in ${Thresholded} \
		 -erode \
		 -dest $TMP \
		 -kernel $kernel \
		 -v;
	else
		${Morpho} \
		 -in ${Thresholded} \
		 -erode \
		 -dest $TMP \
		 -kernel $kernel;
	fi

	checkExitCode $? "\nERROR: Opening of the thresholded mask failed!!!"

	echoV "\n----->	Clustering the eroded mask to remove small, isolated clusters ...\n"
	thresh=$(( `3dBrickStat -slow -non-zero -count ${Sum_ero}` / 2 ))
	
	echoV "--> 3dclust \n
	  -prefix ${Sum_ero_clust} \n
	  0 \n
	  ${thresh} \n
	  ${Sum_ero};"

	if [ "$verbose" == "1" ]
	then
		3dclust \
		 -prefix ${Sum_ero_clust} \
		 0 \
		 ${thresh} \
		 ${Sum_ero};
	else
		3dclust \
		 -summarize \
		 -quiet \
		 -nosum \
		 -prefix ${Sum_ero_clust} \
		 0 \
		 ${thresh} \
		 ${Sum_ero};
	fi
	checkExitCode $? "\nERROR: Clustering of the processed ventricle mask failed!!!"

	echoV "\n----->	Dilating the clustered mask ...\n"
	echoV "${Morpho} \n
	 -in ${Sum_ero_clust} \n
	 -dilate \n
	 -dest $TMP \n
	 -kernel $kernel \n
	 -v;"

	if [ "$verbose" == "1" ]
	then
		${Morpho} \
		 -in ${Sum_ero_clust} \
		 -dilate \
		 -dest $TMP \
		 -kernel $kernel \
		 -v;
	else
		${Morpho} \
		 -in ${Sum_ero_clust} \
		 -dilate \
		 -dest $TMP \
		 -kernel $kernel;
	fi

	checkExitCode $? "\nERROR: Opening of the thresholded mask failed!!!"

else
	echoV "\n----->	No opening of the thresholded mask requested ...\n"
	if [ "$verbose" == "1" ]
	then
		cp -v ${Thresholded} ${Sum_open}
	else
		cp ${Thresholded} ${Sum_open}
	fi
fi

### Filling holes
echoV "\n----->	Filling holes in the final brain mask ...\n"
echoV "\n--> ${fillholes} \n
	 -in ${Sum_open} \n
	 -dest $TMP \n
	 -v;"
	
if [ "$verbose" == "1" ]
then
	${fillholes} \
	 -in ${Sum_open} \
	 -dest $TMP \
	 -v;
else
	${fillholes} \
	 -in ${Sum_open} \
	 -dest $TMP;
fi
checkExitCode $? "\nERROR: Hole Filling failed!!!"

### Clustering the final mask to exclude small, isolated clusters
echoV "\n----->	Clustering the final, threholded, eroded and dilated mask to remove small, isolated clusters ...\n"
thresh=$(( `3dBrickStat -slow -non-zero -count ${Filled}` / 2 ))

echoV "--> 3dclust \n
  -prefix ${Clustered} \n
  0 \n
  ${thresh} \n
  ${Filled};"

if [ "$verbose" == "1" ]
then
	3dclust \
	 -prefix ${Clustered} \
	 0 \
	 ${thresh} \
	 ${Filled};
else
	3dclust \
	 -summarize \
	 -quiet \
	 -nosum \
	 -prefix ${Clustered} \
	 0 \
	 ${thresh} \
	 ${Filled};
fi
checkExitCode $? "\nERROR: Clustering of the processed ventricle mask failed!!!"

### Renaming the final mask
echoV "\n----->	Converting the final CBQ mask to byte ...\n"
if [ "$verbose" == "1" ]
then
#	mv -v ${Clustered} ${mask}.nii.gz
	echoV "\n--> 3dcalc \n
	 -a ${Clustered} \n
	 -prefix ${mask}.nii.gz \n
	 -expr a \n
	 -verbose \n
	 -nscale \n
	 -byte;"

	3dcalc \
	 -a ${Clustered} \
	 -prefix ${mask}.nii.gz \
	 -expr a \
	 -verbose \
	 -nscale \
	 -byte;
else
#	mv ${Clustered} ${mask}.nii.gz
	3dcalc \
	 -a ${Clustered} \
	 -prefix ${mask}.nii.gz \
	 -expr a \
	 -nscale \
	 -byte;
fi

### Removing the Skull and cerebellum
if [ "$vnmask" == "0" ]
then
	echoV "\n----->	Removing the Skull and cerebellum ...\n"
	echoV "\n--> 3dcalc \n
	 -prefix ${cbq}.nii.gz \n
	 -a ${InbName}.nii.gz \n
	 -b ${mask}.nii.gz \n
	 -expr 'a*b' \n
	 -nscale \n
	 -verbose;"

	if [ "$verbose" == "1" ]
	then
		3dcalc \
		 -prefix ${cbq}.nii.gz \
		 -a ${InbName}.nii.gz \
		 -b ${mask}.nii.gz \
		 -expr 'a*b' \
		 -nscale \
		 -verbose;
	else
		3dcalc \
		 -prefix ${cbq}.nii.gz \
		 -a ${InbName}.nii.gz \
		 -b ${mask}.nii.gz \
		 -expr 'a*b' \
		 -nscale;
	fi
	checkExitCode $? "\nERROR: Masking out of skull and cerebellum failed!!!"
fi

### Transferring the results to the destination
echoV "\n----->	Transferring the results to the destination ...\n"

if [ "$dest" != "$TMP" ]
then
	if [ ! -d $dest ]
	then
		if [ "$verbose" == "1" ]
		then
			mkdir -pv $dest
		else
			mkdir -p $dest
		fi
	fi

	if [ "$verbose" == "1" ]
	then
		if [ "$vnmask" == "0" ]
		then
			mv -v ${TMP}${cbq}.nii.gz ${dest}${cbq}.nii.gz
		fi
		mv -v ${TMP}${mask}.nii.gz ${dest}${mask}.nii.gz
	else
		if [ "$vnmask" == "0" ]
		then
			mv ${TMP}${cbq}.nii.gz ${dest}${cbq}.nii.gz
		fi
		mv ${TMP}${mask}.nii.gz ${dest}${mask}.nii.gz
	fi

	### Removing temporary files from the destination
	echoV "\n----->	Removing temporary files from the TMPDIR ...\n"
	if [ "$verbose" == "1" ]
	then
		rm -fv ${TMP}*
		rmdir -v ${TMP}
	else
		rm -f ${TMP}*
		rmdir ${TMP}
	fi
else
	rmV ${Thresholded}
	rmV ${Sum_open}
	rmV ${Filled}
	rmV ${Clustered}
	if [ "$CpInput" == "1" ]
	then
		rmV ${InbName}.nii.gz
	fi
	
	if [ "$CpJacob" == "1" ]
	then
		rmV ${JRbName}.nii.gz
	fi
fi

### Execution Time 
executionTime

################################################ END ################################################
