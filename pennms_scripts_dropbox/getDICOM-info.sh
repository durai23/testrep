#!/usr/bin/env bash

usage()
{
cat << EOF
usage: $0 options

This script extracts a variety of information from a dicom and stores it in a csv file

OPTIONS:
	-h show this message
	-i absolute path to dicom file (can be .bz2)
	-o output directory (optional. default will just print to stdout)
	-s subject ID as first column in output csv (optional)

EOF
}


dFile=

while getopts "hi:o:s:" OPTION
do
	case $OPTION in
		h)
			usage
			exit 1
			;;
		i)
			dFile=$OPTARG
			;;
		o)
			OUTDIR=$OPTARG
			;;
		s)	SUBID=$OPTARG
			;;
		?)
			usage
			exit
			;;
	esac
done

useVerbose=1
bz2Flag=0

if [ -z ${dFile} ] 
then
	usage
	exit 1
fi

if [ ! -n "${OUTDIR}" ]
then
	useVerbose=0
fi

#check if dicoms are compressed 
if [ "${dFile##*.}" = "bz2" ]
then
	bz2Flag=1
        if [ ! ${useVerbose} -eq 0 ]
        then
                echo "Dicoms detected to be bz2 compressed."
        fi
fi





echo "Dicom File:" ${dFile}
echo "Output Dir:" ${OUTDIR}
echo "Subject ID:" ${SUBID}

#if bz2, copy dicom to tempDir
if [ ${bz2Flag} -eq 1 ]
then
	PID=$$
	TMP=${SBIA_TMPDIR}/${PID}
	mkdir ${TMP}
	if [ ! ${useVerbose}  -eq 0 ]
	then
	        echo "Temp Dir Created at:" ${TMP}
	fi

else
	TMP=`dirname ${dFile}`
fi

echo "Working Dir:" ${TMP}
dTemp=${TMP}/${dFile##*/} #dicom file to read info from


#If dicoms have been bz2 compressed, copy them to temp directory and decompress
if [ ${bz2Flag} -eq 1 ]
then
	if [ ! ${useVerbose} -eq 0 ]
	then
		echo "Copying dicoms to ${dTemp}"
	fi

	#copy dicoms
	cp ${dFile%/*}/* ${TMP}
	
	for i in `ls -1 ${TMP}/*`
	do
		bunzip2 ${i}
	done
	dTemp=${TMP}/${dFile##*/}
	dTemp=${dTemp%.bz2}
fi


echo "file to search"
echo ${dTemp}
fd=`ls -1 ${dTemp}`
if [ -z ${fd} ]
then
	echo "not found"
	exit
fi

#read group 008
studyDate_raw=`dcmdump --search 0008,0020 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
seriesDate_raw=`dcmdump --search 0008,0021 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
acqDate_raw=`dcmdump --search 0008,0022 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
accNum=`dcmdump --search 0008,0050 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
manufact=`dcmdump --search 0008,0070 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
stationName=`dcmdump --search 0008,1010 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
Modality=`dcmdump --search 0008,103E ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available" | sed s/' '/_/g`
manufact_modelName=`dcmdump --search 0008,1090 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`

#reformat dates
tempYR=`echo ${studyDate_raw} | cut -c1-4`
tempDY=`echo ${studyDate_raw} | cut -c7-8`
tempMN=`echo ${studyDate_raw} | cut -c5-6`
studyDate=`echo ${tempMN}/${tempDY}/${tempYR}`

tempYR=`echo ${seriesDate_raw} | cut -c1-4`
tempDY=`echo ${seriesDate_raw} | cut -c7-8`
tempMN=`echo ${seriesDate_raw} | cut -c5-6`
seriesDate=`echo ${tempMN}/${tempDY}/${tempYR}`

tempYR=`echo ${acqDate_raw} | cut -c1-4`
tempDY=`echo ${acqDate_raw} | cut -c7-8`
tempMN=`echo ${acqDate_raw} | cut -c5-6`
acqDate=`echo ${tempMN}/${tempDY}/${tempYR}`

echo ACQDate: ${acqDate}
echo SeriesDate: ${seriesDate}
echo StudyDate: ${studyDate}
echo Manufacturer: ${manufact}
echo StationName: ${stationName}
echo Modality: ${Modality}
echo Manufacturer Model Name: ${manufact_modelName}


#read group 0010
name=`dcmdump --search 0010,0010 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
patID=`dcmdump --search 0010,0020 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
dob_raw=`dcmdump --search 0010,0030 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
sex=`dcmdump --search 0010,0040 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
age=`dcmdump --search 0010,1010 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
weight=`dcmdump --search 0010,1030 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`

#reformat DOB
tempYR=`echo ${dob_raw} | cut -c1-4`
tempDY=`echo ${dob_raw} | cut -c7-8`
tempMN=`echo ${dob_raw} | cut -c5-6`
dob=`echo ${tempMN}/${tempDY}/${tempYR}`

echo Name: ${name}
echo PatientID: ${patID}
echo DateOfBirth: ${dob}
echo Sex: ${sex}
echo Age: ${age}
echo Weight: ${weight}


#read group 0018
scan_seq=`dcmdump --search 0018,0020 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
acq_type=`dcmdump --search 0018,0023 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
slice_thickness=`dcmdump --search 0018,0050 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
rep_time=`dcmdump --search 0018,0080 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
echo_time=`dcmdump --search 0018,0081 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
inv_time=`dcmdump --search 0018,0082 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
img_freq=`dcmdump --search 0018,0084 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
mag_strength=`dcmdump --search 0018,0087 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
slice_spacing=`dcmdump --search 0018,0088 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
flip_angle=`dcmdump --search 0018,1314 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
sar=`dcmdump --search 0018,1316 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"` #specific absorption rate
#nx=`dcmdump --search 0018,1310 ${dTemp} | cut -d' ' -f3 | tr '\\' ' ' | tr '0' ' ' | tr -s ' ' | sed s/^' '//g | cut -d' ' -f1` 
#ny=`dcmdump --search 0018,1310 ${dTemp} | cut -d' ' -f3 | tr '\\' ' ' | tr '0' ' ' | tr -s ' ' | sed s/^' '//g | cut -d' ' -f2` 

nx=$( for i in `dcmdump --search 0018,1310 ${dTemp} | cut -d' ' -f3 | tr '\\' ' '`
do 
	echo $i
done | grep -v -w "0" | head -n 1 )

ny=$( for i in `dcmdump --search 0018,1310 ${dTemp} | cut -d' ' -f3 | tr '\\' ' '`
do 
	echo $i
done | grep -v -w "0" | tail -n 1 )


echo ScanSequence: ${scan_seq}
echo AcquisitionType: ${acq_type}
echo SliceThickness: ${slice_thickness}
echo RepetitionTime: ${rep_time}
echo EchoTime: ${echo_time}
echo InversionTime: ${inv_time}
echo ImagingFreq: ${img_freq}
echo MagneticFieldStrength: ${mag_strength}
echo SliceSpacing: ${slice_spacing}
echo FlipAngle: ${flip_angle}
echo SpecificAbsorptionRate: ${sar}
echo AcqMatRows: ${nx}
echo AcqMatCols: ${ny}

#read group 0020
rel_img_pos_patient=`dcmdump --search 0020,0032 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
rel_img_ori=`dcmdump --search 0020,0037 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available" | cut -d\# -f1`

#find maximum series number to determine nz
for i in `ls -1 ${TMP}/*`
do
	dcmdump --search 0020,0013 ${i} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available" >> ${TMP}/rel_img_series.csv
done

nz=`cat ${TMP}/rel_img_series.csv | sort -n | tail -n 1`
rm -fv ${TMP}/rel_img_series.csv


#read group 0028
rows=`dcmdump --search 0028,0010 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
cols=`dcmdump --search 0028,0011 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
pixel_spacing=`dcmdump --search 0028,0030 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
bits_alloc=`dcmdump --search 0028,0100 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
bits_stored=`dcmdump --search 0028,0101 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
win_center=`dcmdump --search 0028,1050 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`
win_width=`dcmdump --search 0028,1051 ${dTemp} | cut -d[ -f2 | cut -d] -f1 | grep -v "no value available"`

#check if more than one set of image dimensions found
n=`echo ${rows} | grep \# `
if [ -n "${n}" ]
then
	rows=`echo ${rows} | cut -d\# -f2 | cut -d' ' -f7`
fi

n=`echo ${cols} | grep \# `
if [ -n "${n}" ]
then
	cols=`echo ${cols} | cut -d\# -f2 | cut -d' ' -f7`
fi

n=`echo ${bits_alloc} | grep \# `
if [ -n "${n}" ]
then
	bits_alloc=`echo ${bits_alloc} | cut -d\# -f2 | cut -d' ' -f7`
fi

n=`echo ${bits_stored} | grep \# `
if [ -n "${n}" ]
then
	bits_stored=`echo ${bits_stored} | cut -d\# -f2 | cut -d' ' -f7`
fi

n=`echo ${win_center} | wc -w`
if [ ${n} -eq 2 ]
then
	win_center=`echo ${win_center} | cut -d' ' -f2 `
	win_width=`echo ${win_width} | cut -d' ' -f2 `
fi
if [ ${n} -gt 2 ]
then
	echo ERROR: more than two sets of pixeldims found
fi




echo RelativeImagePositionPatient: ${rel_img_pos_patient}
echo RelativeImageOrientation: ${rel_img_ori}
echo Rows: ${rows}
echo Cols: ${cols}
echo Slices: ${nz}
echo PixDim: ${pixel_spacing}
echo BitsAllocated: ${bits_alloc}
echo BitsStored: ${bits_stored}
echo WindowCenter: ${win_center}
echo WindowWidth: ${win_width}


if [ ! -z ${OUTDIR} ]
then

	#Write Data to csv file
	echo "ID,Name,PatientID,Modality,AccessionNum,ACQDate,StudyDate,SeriesDate,Manufacturer,StationName,ManufacturerModelName,"\
	"DateOfBirth,Sex,Age,Weight,ScanSequence,AcquisitionType,SliceThickness,RepetitionTime,EchoTime,InversionTime,"\
	"ImagingFreq,MagneticFieldStrength,SliceSpacing,FlipAngle,SpecificAbsorptionRate,"\
	"Rows,Cols,Slices,AcqMatRows,AcqMatCols,PixDim,BitsAllocated,BitsStored,dcmPATH" >> ${OUTDIR}/${SUBID}_dicomInfo_raw.csv
	echo ${SUBID},${name},${patID},${Modality},${accNum},${acqDate},${studyDate},${seriesDate},${manufact},${stationName},${manufact_modelName},\
	${dob},${sex},${age},${weight},${scan_seq},${acq_type},${slice_thickness},${rep_time},${echo_time},${inv_time},\
	${img_freq},${mag_strength},${slice_spacing},${flip_angle},${sar},${rows},\
	${cols},${nz},${nx},${ny},${pixel_spacing},${bits_alloc},${bits_stored},${dFile%/*} >> ${OUTDIR}/${SUBID}_dicomInfo_raw.csv
	
fi

if [ ${bz2Flag} -eq 1 ]
then
	rm -rf ${TMP}
fi



