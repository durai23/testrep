#!/usr/bin/env bash

sub=$1
dcmFile=$2 #path to input dicomInfo csv file
output=$3 #path to output csv file

if [ ! -e ${dcmFile} ]
then
	echo "dicomInfo file not found for " ${sub}
	exit
fi

#make header
echo "NiftiID",`cat ${dcmFile} | head -n 1` > ${output}

#extract t1
n=`cat ${dcmFile} | cut -d, -f4 | grep -n -i "t1" | grep -v -i "post" | grep -v -i "dynasuite" | grep -v -i "bodycoil" | cut -d: -f1`
if [ ! `echo ${n} | wc -l` -eq 0 ]
then
   
	wc=`echo ${n} | wc -w`
	if [ ${wc} -eq 1 ]
	then
		l=`sed -n ${n}p ${dcmFile} `
                echo ${sub}_t1,${l} >> ${output}
	else	 
		count=1;	
		for j in `echo ${n}`
	       	do 
			l=`sed -n ${j}p ${dcmFile} `
			echo ${sub}_t1_${count},${l} >> ${output}
			count=$(( count + 1 ))
	       	done
	fi
fi

#extract t2
n=`cat ${dcmFile} | cut -d, -f4 | grep -n -i "t2" | grep -v -i "flair" | grep -v -i "dynasuite" | cut -d: -f1`
if [ ! `echo ${n} | wc -l` -eq 0 ]
then 

	wc=`echo ${n} | wc -w`
        if [ ${wc} -eq 1 ]
        then
                l=`sed -n ${n}p ${dcmFile} `
                echo ${sub}_t2,${l} >> ${output}
        else
                count=1;
                for j in `echo ${n}`
                do
                        l=`sed -n ${j}p ${dcmFile} `
                        echo ${sub}_t2_${count},${l} >> ${output}
                        count=$(( count + 1 ))
                done
        fi


fi

#extract t1-gad
n=`cat ${dcmFile} | cut -d, -f4 | grep -E -n -i "post|gad" | grep -v -i "dynasuite" | grep -v -i "flair" | cut -d: -f1`
if [ ! `echo ${n} | wc -l` -eq 0 ]
then 

	wc=`echo ${n} | wc -w`
        if [ ${wc} -eq 1 ]
        then
                l=`sed -n ${n}p ${dcmFile} `
                echo ${sub}_t1ce,${l} >> ${output}
        else
                count=1;
                for j in `echo ${n}`
                do
                        l=`sed -n ${j}p ${dcmFile} `
                        echo ${sub}_t1ce_${count},${l} >> ${output}
                        count=$(( count + 1 ))
                done
        fi


fi

#extract flair
n=`cat ${dcmFile} | cut -d, -f4 | grep -n -i "flair" | grep -v -i "dynasuite" | cut -d: -f1`
if [ ! `echo ${n} | wc -l` -eq 0 ]
then 

	wc=`echo ${n} | wc -w`
        if [ ${wc} -eq 1 ]
        then
                l=`sed -n ${n}p ${dcmFile} `
                echo ${sub}_flair,${l} >> ${output}
        else
                count=1;
                for j in `echo ${n}`
                do
                        l=`sed -n ${j}p ${dcmFile} `
                        echo ${sub}_flair_${count},${l} >> ${output}
                        count=$(( count + 1 ))
                done
        fi


fi


#extract dti
n=`cat ${dcmFile} | cut -d, -f4 | grep -n -E -i "dti|dwi|diff" | grep -v -i "dynasuite" | grep -E -v -i "trace|fa|adc|tensor" | cut -d: -f1`
if [ ! `echo ${n} | wc -l` -eq 0 ]
then 

	wc=`echo ${n} | wc -w`
        if [ ${wc} -eq 1 ]
        then
                l=`sed -n ${n}p ${dcmFile} `
                echo ${sub}_dti,${l} >> ${output}
        else
                count=1;
                for j in `echo ${n}`
                do
                        l=`sed -n ${j}p ${dcmFile} `
                        echo ${sub}_dti_${count},${l} >> ${output}
                        count=$(( count + 1 ))
                done
        fi


fi

