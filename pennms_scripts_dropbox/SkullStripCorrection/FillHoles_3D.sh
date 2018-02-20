#!/bin/bash

################################################ VERSION INFO ################################################
# $Id: FillHoles_3D.sh 79 2012-12-04 21:05:33Z doshijim@UPHS.PENNHEALTH.PRV $
#
version()
{
	# Display the version number and date of modification extracted from
	# the Id variable.
	SVNversion="$Id: FillHoles_3D.sh 79 2012-12-04 21:05:33Z doshijim@UPHS.PENNHEALTH.PRV $"
	Auth="Jimit Doshi"
	ver="$Rev: 79 $"
	mod="$LastChangedDate: 2012-12-04 16:05:33 -0500 (Tue, 04 Dec 2012) $"
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

Reqd:	-in	   < path >	absolute path of the input file to be registered
	
Opt:	-dest	   < path >	absolute path to the destination where the results are to be stored (default: same as input)
	-pref	   < pattern >	output file name (default: input_filled)
	-tmp	   < path >	absolute path to the temporary directory (default: \$SBIA_TMPDIR )
	-v	   		verbose output (default: 0 - no output)


ERROR: Not enough arguments!!
##############################################

DEPENDENCIES:
	3dcalc		: `which 3dcalc`
	3dBrickStat	: `which 3dBrickStat`
	nifti1_test	: `which nifti1_test`
	3dclust		: `which 3dclust`
	
HELP
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
		rm -rfv ${TMP}*
	fi
	
	exit 1
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
			-pref) 
				pref=$2;
				shift 2;;			# source path is set
			-v) 
				verbose=1;
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

echoV()
{
	if [ "$verbose" == "1" ]
	then
		echo -e $1
	fi
}

import()
{
	ext=$1
	inFile=$2
	outFile=$3
	
	if [ "${ext}" == "nii.gz" ]
	then
		if [ "$verbose" == "1" ]
		then
			cp -v ${inFile} ${outFile}.nii.gz
		else
			cp ${inFile} ${outFile}.nii.gz
		fi			
	elif [ "${ext}" == "nii" ]
	then
		echoV "nifti1_test -zn1 ${inFile} ${outFile}"
		nifti1_test -zn1 ${inFile} ${outFile}
	elif [ "${ext}" == "hdr" ]
	then
		echoV "nifti1_test -zn1 ${inFile%.hdr}.img ${outFile}"
		nifti1_test -zn1 ${inFile%.hdr}.img ${outFile}
	elif [ "${ext}" == "img" ]
	then
		echoV "nifti1_test -zn1 ${inFile} ${outFile}"
		nifti1_test -zn1 ${inFile} ${outFile}
	fi	
}

################################################ END OF FUNCTIONS ################################################

################################################ MAIN BODY ################################################

if [ $# -lt 2 ]; then
	help
fi

### Timestamps
startTime=`date +%F-%H:%M:%S`
startTimeStamp=`date +%s`

### Specifying the trap signal
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGHUP signal'" SIGHUP 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGINT signal'" SIGINT 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGTERM signal'" SIGTERM 
trap "checkExitCode 1 '\nProgram Interrupted. Received SIGKILL signal'" SIGKILL

### Reading the arguments
parse $*

echoV "\nRunning commands on		: `hostname`"
echoV "Start time			: ${startTime}\n"

### Check if all dependenices are satisfied
checkDependency	3dcalc
checkDependency	nifti1_test
checkDependency	3dBrickStat
checkDependency	3dclust

### Forming FileNames
# TMP
PID=$$

if [ -n "$tmp" ]
then
	if [ ! -d "$tmp" ]
	then
		mkdir -p $tmp
	fi

	TMP=`mktemp -d -p ${tmp} FillHoles_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; TMP=${tmp}/FillHoles_${PID}; mkdir -pv ${TMP}; }
elif [ -n "$SBIA_TMPDIR" ]
then
	if [ ! -d "$SBIA_TMPDIR" ]
	then
		mkdir -p $SBIA_TMPDIR
	fi

	TMP=`mktemp -d -p ${SBIA_TMPDIR} FillHoles_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; TMP=${SBIA_TMPDIR}/FillHoles_${PID}; mkdir -pv ${TMP}; }
else
	TMP=`mktemp -d -t FillHoles_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; exit 1; }
fi
echoV "\n----->	Temporary local directory created at $TMP ...\n"

### Checking for default parameters
if [ -z $dest ]
then
	dest=`dirname $input`/
	if [ "$dest" == "./" ]
	then
		dest=`pwd`/
	fi
fi

if [ -z $pref ]
then
	pref=${InbName}_filled
fi

if [ "$verbose" == "1" ]
then
	echo -e "Input Image			: ${input}"
	echo -e "Output Image			: ${dest}${pref}.nii.gz"
fi

### Importing data to the temporary directory
echoV "\n----->	Importing required files to the temporary local directory ...\n"
import ${InExt} ${input} ${TMP}${InbName}

cd $TMP

### Converting the Input image to a binary mask (if it is not already) and inverting the mask
echoV "\n----->	Converting the Input image to a binary mask (if it is not already) and inverting the mask ...\n"
echoV "\n--> 3dcalc \n
	 -prefix ${TMP}${InbName}_inv.nii.gz \n
	 -a ${TMP}${InbName}.nii.gz \n
	 -expr 'iszero(step(a))' \n
	 -verbose \n
	 -nscale \n
	 -byte;"
	
if [ "$verbose" == "1" ]
then
	3dcalc \
	 -prefix ${TMP}${InbName}_inv.nii.gz \
	 -a ${TMP}${InbName}.nii.gz \
	 -expr 'iszero(step(a))' \
	 -verbose \
	 -nscale \
	 -byte;
else
	3dcalc \
	 -prefix ${TMP}${InbName}_inv.nii.gz \
	 -a ${TMP}${InbName}.nii.gz \
	 -expr 'iszero(step(a))' \
	 -nscale \
	 -byte >> FillHoles.log 2>&1
fi
checkExitCode $? "\nERROR: Binarizing and Inversion of the Input image failed!!!"

### Clusterizing the inverted mask
vol=`3dBrickStat -slow -non-zero -count ${TMP}${InbName}_inv.nii.gz`
thresh=`echo "scale=0; $vol / 2" | bc`

echoV "\n----->	Clustering the inverted mask ...\n"
echoV "\n--> 3dclust \n
	 -prefix ${TMP}${InbName}_inv_bg.nii.gz \n
	 0 ${thresh} \n
	 ${TMP}${InbName}_inv.nii.gz; "
	
if [ "$verbose" == "1" ]
then
	3dclust \
	 -prefix ${TMP}${InbName}_inv_bg.nii.gz \
	 0 ${thresh} \
	 ${TMP}${InbName}_inv.nii.gz; 
else
	3dclust \
	 -prefix ${TMP}${InbName}_inv_bg.nii.gz \
	 0 ${thresh} \
	 ${TMP}${InbName}_inv.nii.gz  >> FillHoles.log 2>&1
fi
checkExitCode $? "\nERROR: Clusterizing failed!!!"

### Clusterizing the inverted mask
echoV "\n----->	Creating the final mask ...\n"
echoV "\n--> 3dcalc \n
	 -prefix ${TMP}${pref}.nii.gz \n
	 -a ${TMP}${InbName}.nii.gz \n
	 -b ${TMP}${InbName}_inv.nii.gz \n
	 -c ${TMP}${InbName}_inv_bg.nii.gz \n
	 -expr 'a+(b-c)' \n
	 -verbose \n
	 -nscale;"
	
if [ "$verbose" == "1" ]
then
	3dcalc \
	 -prefix ${TMP}${pref}.nii.gz \
	 -a ${TMP}${InbName}.nii.gz \
	 -b ${TMP}${InbName}_inv.nii.gz \
	 -c ${TMP}${InbName}_inv_bg.nii.gz \
	 -expr 'a+(b-c)' \
	 -verbose \
	 -nscale; 
else
	3dcalc \
	 -prefix ${TMP}${pref}.nii.gz \
	 -a ${TMP}${InbName}.nii.gz \
	 -b ${TMP}${InbName}_inv.nii.gz \
	 -c ${TMP}${InbName}_inv_bg.nii.gz \
	 -expr 'a+(b-c)' \
	 -verbose \
	 -nscale  >> FillHoles.log 2>&1
fi
checkExitCode $? "\nERROR: Filling Holes failed!!!"

### Transferring the results to the destination
echoV "\n----->	Transferring the results to the destination ...\n"
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
	mv -v ${TMP}${pref}.nii.gz ${dest}
else
	mv ${TMP}${pref}.nii.gz ${dest}
fi

### Removing the remaining files
echoV "\n----->	Removing some of the remaining files ...\n"

if [ "$verbose" == "1" ]
then
	rm -fv ${TMP}${InbName}.nii.gz
	rm -fv ${TMP}${InbName}_inv.nii.gz
	rm -fv ${TMP}${InbName}_inv_bg.nii.gz
	rm -fv FillHoles.log
	rmdir -v $TMP
else
	rm -f ${TMP}${InbName}.nii.gz
	rm -f ${TMP}${InbName}_inv.nii.gz
	rm -f ${TMP}${InbName}_inv_bg.nii.gz
	rm -f FillHoles.log
	rmdir $TMP 
fi	


################################################ END ################################################

### Execution Time 
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
