#!/bin/bash
rm -rf ./pseudonym_map.csv
infile=$1
echo "NAME,RNDNAME,SNO" >> ./pseudonym_map.csv
COUNTER=1
for id in `cat $infile | cut -d, -f1`
do
	echo "Serial No: $COUNTER"
	echo "for ID: $id"
	RANDOM=$id
	#generate random number between 1 and x
	rndid=`echo $((999 + RANDOM % 999999))`
	echo "for ID random: $rndid"
	echo "$id,$rndid,$COUNTER" >> ./pseudonym_map.csv
	COUNTER=$[$COUNTER +1]
done

#chekc if no random repeats

