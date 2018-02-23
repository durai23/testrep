#!/bin/sh

. /sbia/home/doshijim/General_Scripts/BashUtilityFunctions.sh
. /usr/share/Modules/init/bash
if [ $? != 0 ]
then
        echo "Failure to load /usr/share/Modules/init/bash"
        exit 1
fi

module unload python; 
module load python/2.5.2; 

################################################ FUNCTIONS ################################################

help()
{
cat <<HELP

This script does the following:

Files required:

Tools used:

##############################################
Usage :	$0 [OPTIONS]
OPTIONS:
Reqd:	-in	< file >	absolute path to the input file to be skull-stripped
	 
Opt:	-h         		this help page
	-dest	< path >	absolute path to the destination where the results are to be stored (default: same as input)
	-pref	< file >	prefix of the output file (default: input_n3)

ERROR: Not enough arguments!!

HELP
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
				checkFile $input; 

				temp=`FileAtt $input`				
				Inext=`echo $temp | awk '{ print $1 }'`
				InbName=`echo $temp | awk '{ print $2 }'`

				shift 2;;			# SubID is set
			-pref) 
				pref=$2;
				shift 2;;			# source path is set
			-dest) 
				dest=$2;
				shift 2;;			# source path is set
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
	nifti1_test -zn1 $1 $1

	if [ -f ${1%.img}.nii.gz ]
	then
		echo -e "\nConverted to NIFTIGZ: $1"
		rm -fv ${1} ${1%.img}.hdr
	else
		echo -e "\nConversion to NIFTIGZ failed: $1"
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

echo -e "\nRunning commands on		: `hostname`"
echo -e "Start time			: ${startTime}\n"

### Reading the arguments
echo -e "Parsing arguments		: $*"
parse $*

### Forming FileNames
PID=$$
TMP=${SBIA_TMPDIR}/BiasCorrect_${PID}/

OutputImage=${TMP}${InbName}_n3

### Checking if destination was provided, if not, same as input file
if [ -z "$dest" ]
then
	dest=`dirname $input`/
fi

if [ -z "$pref" ]
then
	pref=${InbName}_n3
fi

if [ ! -f ${dest}${pref}.nii.gz ]
then
	### Echoeing filenames
	echo -e "\nINPUT FILES"
	echo "Input Image			: ${input}"

	echo -e "\nOUTPUT FILES"
	echo "Skull-stripped image		: ${dest}${pref}.nii.gz"

	### Creating temporary directory
	echo -e "\n----->	Creating temporary local directory ...\n"
	mkdir -pv $TMP

	### Importing data to the temporary directory
	echo -e "\n----->	Importing required files to the temporary local directory ...\n"

	if [ "$Inext" == "hdr" ]
	then
		cp -v ${input} ${TMP}${InbName}.hdr
		cp -v ${input%.hdr}.img ${TMP}${InbName}.img
	elif [ "$Inext" == "nii.gz" ]
	then
		echo "nifti1_test -n2 ${input} ${TMP}${InbName}"
		nifti1_test -n2 ${input} ${TMP}${InbName}
	fi

	### Bias Correction
	echo -e "\n----->	Bias Correcting $input ...\n"
	echo -e "n3BiasCorrection.py -d ${TMP}${InbName}.hdr -p ${OutputImage} -v"
	n3BiasCorrection.py -d ${TMP}${InbName}.hdr -p ${OutputImage} -v 

	### Removing temporary files
	echo -e "\n----->	Removing temporary files ...\n"
	rm -fv ${TMP}${InbName}.img ${TMP}${InbName}.hdr

	### Converting to NIFTIGZ
	echo -e "\n----->	Converting the results to NIFTIGZ file format ...\n"
	convertToNifti ${OutputImage}.img ${OutputImage}

	### Moving the results to the destination directory
	# Checking if the destination exists, if not, create it
	echo -e "\n----->	Moving the results to the destination directory ...\n"
	if [ ! -d ${dest} ]
	then
		mkdir -pv $dest
	fi

	mv -v ${TMP}* ${dest}

	rmdir -pv $TMP
else
	echo -e "\n\nResults already exist!"
fi

################################################ END ################################################

endTimeStamp=`date +%s`;
echo -e "\nExecution time:  $[ (${endTimeStamp} - ${startTimeStamp}) / 60]m $[ (${endTimeStamp} - ${startTimeStamp}) % 60]s"