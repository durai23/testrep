#prints message with date and time to csv file
#to add new log edit --lg line
import time
import os
import csv
import argparse

def addlog(fta,entry):
	dt=time.strftime("%x")
	tm=time.strftime("%X")
	with open(fta, 'ab') as fp:
		w = csv.writer(fp,dialect='excel')
		data = [[dt,tm,entry]] #inside square braces reqd or it writes each char in separate cell
		w.writerows(data)
		fp.close()

parser=argparse.ArgumentParser(description='get log and msg')
#make it easy to add further logs - both here and at file naming
parser.add_argument('entry', nargs='*')
parser.add_argument('--lg', choices=['log', 'aud', 'wthr', 'momb', 'dadb', 'adp', 'cds', 'trfc', 'nrop', 'mov', 'yt', 'isp', 'escl', 'sem', 'sab', 'reg', 'crf', '2crf', 'phole'], required=True)
args=parser.parse_args()

msg=' '.join(args.entry)

#print 'message to log is '+msg

#now=time.strftime("%c")
#print '%s' % now
diry='/localdata/dnambi/testrep/pylogger/'+time.strftime('%m')+'_'+time.strftime('%d')+'_'+time.strftime('%Y')
fily='/localdata/dnambi/testrep/pylogger/'+time.strftime('%m')+'_'+time.strftime('%d')+'_'+time.strftime('%Y')+'/'+args.lg+'.csv'
#print diry
#print fily
#if os.path.isfile('C:\Users\synapse\Desktop\downloads\s3')
#creates folder and file if they dont exist
if(os.path.exists(diry)):
	addlog(fily,msg)
else:
	os.makedirs(diry)
	print 'made dir for today. appending...'
	addlog(fily,msg)
	#could set flag, not doing as no shortage of RAM and ONE binary check is inevitable
    #w = csv.writer(open(Fn,'ab'),dialect='excel')

# use-case
# DOES APPEND CREATE FILE?
# 	dir_creation
# 		manual
# 			dir already exists
# 	    through logger
# 	    	then check if dir exists first
# 	    		no 
# 	    			create dir, create file
# 	    		yes (subseq logs)

# 	    			call append
# 	    				so write yes part first as this use-case used more often


#check if dir exists
# 	no 
# 		create dir,create file
# 			call append
# 	yes
# 		check if file exists
# 			yes
# 				call append
# 			no
# 				create file
# 					call append

# #check if file 
# 	yes 
# 		call append
# 	no 
# 		check if dir
# 			yes
# 				create file
# 					call append
# 			no
# 				create dir,create file
# 						call append
