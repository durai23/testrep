#!/bin/bash

################################################ VERSION INFO ################################################
# $Id: Morpho.sh 79 2012-12-04 21:05:33Z doshijim@UPHS.PENNHEALTH.PRV $
#
version()
{
	# Display the version number and date of modification extracted from
	# the Id variable.
	SVNversion="$Id: Morpho.sh 79 2012-12-04 21:05:33Z doshijim@UPHS.PENNHEALTH.PRV $"
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

Required:	-in	   < path >	absolute path of the input file to be dilated
 
Operation:	-dilate			Perform a dilation operation using a Gaussian sphere (output prefix: input_dil\${kernel}mm)
		-erode			Perform an erosion operation using a Gaussian sphere (output prefix: input_ero\${kernel}mm)
		-open			Perform an opening operation using a Gaussian sphere (output prefix: input_open\${kernel}mm)
		-close			Perform a closing operation using a Gaussian sphere (output prefix: input_close\${kernel}mm)
	
Optional:	-dest	   < path >	absolute path to the destination where the results are to be stored (default: same as input)
		-tmp	   < path >	absolute path to the temporary directory (default: \$SBIA_TMPDIR )
		-kernel	   < int >	Gaussian kernel sphere radius in mm (default: 2)
		-v	   		verbose output (default: 0 - no output)


ERROR: Not enough arguments!!
##############################################

DEPENDENCIES:
	3dcalc		: `which 3dcalc`
	3dBrickStat	: `which 3dBrickStat`
	nifti1_test	: `which nifti1_test`
	3dLocalstat	: `which 3dLocalstat`
	
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
			-kernel) 
				kernel=$2;
				shift 2;;			# source path is set
			-dilate) 
				dil=1;
				shift 1;;			# source path is set
			-erode) 
				ero=1;
				shift 1;;			# source path is set
			-open) 
				open=1;
				shift 1;;			# source path is set
			-close) 
				close=1;
				shift 1;;			# source path is set
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

dilate()
{
	in=$1
	out=$2
	
	
	### Dilating the mask
	echoV "\n----->	Dilating the mask ...\n"
	echoV "3dLocalstat \n
	 -nbhd 'SPHERE($kernel)' \n
	 -stat 'sum' \n
	 -prefix dilate.nii.gz \n
	 ${in};"
	
	if [ "$verbose" == "1" ]
	then 
		3dLocalstat \
		 -nbhd "SPHERE($kernel)" \
		 -stat 'sum' \
		 -prefix dilate.nii.gz \
		 ${in};
	else
		3dLocalstat \
		 -nbhd "SPHERE($kernel)" \
		 -stat 'sum' \
		 -prefix dilate.nii.gz \
		 ${in} >> DilateErode.log 2>&1	
	fi

	echoV "\n3dcalc \n
	 -a dilate.nii.gz \n
	 -expr 'step(a)' \n
	 -prefix ${out} \n
	 -verbose;"

	if [ "$verbose" == "1" ]
	then 
		3dcalc \
		 -a dilate.nii.gz \
		 -expr "step(a)" \
		 -prefix ${out} \
		 -verbose;
	else
		3dcalc \
		 -a dilate.nii.gz \
		 -expr "step(a)" \
		 -prefix ${out} \
		 -verbose >> DilateErode.log 2>&1
	fi

	checkExitCode $? "\nERROR: Dilation of the input image failed!!!"

	if [ "$verbose" == "1" ]
	then
		rm -fv dilate.nii.gz
	else
		rm -f dilate.nii.gz
	fi

}

erode()
{
	in=$1
	out=$2
	
	### Eroding the thresholded mask
	echoV "\n----->	Eroding the thresholded mask ...\n"
	echoV "3dLocalstat \n
	 -nbhd 'SPHERE($kernel)' \n
	 -stat 'sum' \n
	 -prefix erode.nii.gz \n
	 ${in};"
	 
	if [ "$verbose" == "1" ]
	then 
		3dLocalstat \
		 -nbhd "SPHERE($kernel)" \
		 -stat 'sum' \
		 -prefix erode.nii.gz \
		 ${in};
	else
		3dLocalstat \
		 -nbhd "SPHERE($kernel)" \
		 -stat 'sum' \
		 -prefix erode.nii.gz \
		 ${in} >> DilateErode.log 2>&1
	fi
	
	max=$(( `3dBrickStat -slow -max erode.nii.gz` - 1 ))

	echoV "\n3dcalc \n
	 -a erode.nii.gz \n
	 -expr 'step(a-$max)' \n
	 -prefix ${out} \n
	 -verbose;"
	 
	if [ "$verbose" == "1" ]
	then 
		3dcalc \
		 -a erode.nii.gz \
		 -expr "step(a-$max)" \
		 -prefix ${out} \
		 -verbose;
	else
		3dcalc \
		 -a erode.nii.gz \
		 -expr "step(a-$max)" \
		 -prefix ${out} \
		 -verbose >> DilateErode.log 2>&1
	fi	 
	checkExitCode $? "\nERROR: Erosion of the input image failed!!!"

	if [ "$verbose" == "1" ]
	then
		rm -fv erode.nii.gz
	else
		rm -f erode.nii.gz
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

### Default parameters
kernel=2
verbose=0
dil=0
ero=0
open=0
close=0

### Reading the arguments
parse $*

if [ `echo "scale=0; $dil + $ero + $open + $close" | bc` -gt 0 ]
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
else
	checkExitCode 1 "\nERROR: No morphological operation selected!!!"
fi

echoV "\nRunning commands on		: `hostname`"
echoV "Start time			: ${startTime}\n"

### Check if all dependenices are satisfied
checkDependency	3dcalc
checkDependency	nifti1_test
checkDependency	3dBrickStat
checkDependency	3dLocalstat

### Forming FileNames
# TMP
PID=$$

if [ -n "$tmp" ]
then
	if [ ! -d "$tmp" ]
	then
		mkdir -p $tmp
	fi

	TMP=`mktemp -d -p ${tmp} MorphoOperation_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; TMP=${tmp}/MorphoOperation_${PID}; mkdir -pv ${TMP}; }
elif [ -n "$SBIA_TMPDIR" ]
then
	if [ ! -d "$SBIA_TMPDIR" ]
	then
		mkdir -p $SBIA_TMPDIR
	fi

	TMP=`mktemp -d -p ${SBIA_TMPDIR} MorphoOperation_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; TMP=${SBIA_TMPDIR}/MorphoOperation_${PID}; mkdir -pv ${TMP}; }
else
	TMP=`mktemp -d -t MorphoOperation_${PID}.XXXXXXXXXX`/ || { echo -e "\nCreation of Temporary Directory failed."; exit 1; }
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

### Importing data to the temporary directory
echoV "\n----->	Importing required files to the temporary local directory ...\n"
import ${InExt} ${input} ${TMP}${InbName}

cd $TMP

### Converting the Input image to a binary mask (if it is not already) 
echoV "\n----->	Converting the Input image to a binary mask (if it is not already) ...\n"
echoV "\n--> 3dcalc \n
	 -prefix ${TMP}${InbName}_mask.nii.gz \n
	 -a ${TMP}${InbName}.nii.gz \n
	 -expr 'step(a)' \n
	 -verbose \n
	 -nscale \n
	 -byte;"
	
if [ "$verbose" == "1" ]
then
	3dcalc \
	 -prefix ${TMP}${InbName}_mask.nii.gz \
	 -a ${TMP}${InbName}.nii.gz \
	 -expr 'step(a)' \
	 -verbose \
	 -nscale \
	 -byte;
else
	3dcalc \
	 -prefix ${TMP}${InbName}_mask.nii.gz \
	 -a ${TMP}${InbName}.nii.gz \
	 -expr 'step(a)' \
	 -nscale \
	 -byte >> DilateErode.log 2>&1
fi
checkExitCode $? "\nERROR: Binarizing of the Input image failed!!!"

### Performing the morphological operation
if [ "$dil" == "1" ]
then
	dilate ${TMP}${InbName}_mask.nii.gz ${dest}${InbName}_dil${kernel}mm.nii.gz
fi

if [ "$ero" == "1" ]
then
	erode ${TMP}${InbName}_mask.nii.gz ${dest}${InbName}_ero${kernel}mm.nii.gz
fi

if [ "$open" == "1" ]
then
	erode ${TMP}${InbName}_mask.nii.gz ${TMP}${InbName}_ero${kernel}mm.nii.gz
	dilate ${TMP}${InbName}_ero${kernel}mm.nii.gz ${dest}${InbName}_open${kernel}mm.nii.gz
fi

if [ "$close" == "1" ]
then
	dilate ${TMP}${InbName}_mask.nii.gz ${TMP}${InbName}_dil${kernel}mm.nii.gz
	erode ${TMP}${InbName}_dil${kernel}mm.nii.gz ${dest}${InbName}_close${kernel}mm.nii.gz
fi


### Removing the remaining files
echoV "\n----->	Removing some of the remaining files ...\n"

if [ "$verbose" == "1" ]
then
	rm -fv ${TMP}*
	rmdir -v $TMP
else
	rm -f ${TMP}*
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
