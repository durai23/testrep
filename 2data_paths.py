import os
import argparse
import pandas as pd
import time

# printlist=pd.read_csv(r'C:\Users\synapse\Desktop\11_21_2018_data_paths\print_list.csv')

# if printlist.include
# Instantiate the parser
parser = argparse.ArgumentParser(description='path printer')
# Required positional argument
parser.add_argument('--in_folder',help='folder to be printed')
# Required positional argument
parser.add_argument('--out_file',help='output file path printed')
args=parser.parse_args()
print("Argument values:")
print(args.in_folder)
print(args.out_file)

#a=open(r'C:\Users\synapse\Desktop\fzj_1.txt','w')
# a=open(args.out_file,'w')
#for path, subdirs, files in os.walk(r'N:\fzj_temp_dump'):
# for path, subdirs, files in os.walk(args.in_folder):
# 	for filename in files:
# 		f=os.path.join(path,filename)
# 		print f
# 	for levelonedirs in subdirs:
# 		d_one=os.path.join(path,levelonedirs)
# 		print d_one
# THIS WORKS JUST REPLACE args.in_folder with drive letter
# but wont print files in drive immediate base
# THEN add pandas metadata saving
def print_paths_and_log_to_csv(in_dir):
	f_count=0
	cols=['fpath','size','date_created','date_modified']
	in_dir_base=os.path.basename(in_dir)
	out_csv=os.path.join("C:",os.sep,"cygwin64","home","synapse","testrep","data_paths",in_dir_base+".csv")
	out_txt=os.path.join("C:",os.sep,"cygwin64","home","synapse","testrep","data_paths",in_dir_base+".txt")
	df=pd.DataFrame(columns=cols)
	a=open(out_txt,'w')
	for path, subdirs, files in os.walk(in_dir):
		for filename in files:
			f=os.path.join(path,filename)
			fsize=os.path.getsize(f)
			fctime=time.ctime(os.path.getctime(f))
			fmtime=time.ctime(os.path.getmtime(f))
			df1=pd.DataFrame([[f,fsize,fctime,fmtime]],columns=cols)
			#print f
			# pd.concat([df,df1])
			df=df.append(df1,ignore_index=True)
			a.write(str(f)+",")
			f_count+=1
	

	df.to_csv(out_csv)

	print "files processes "+str(f_count)
	# print "5 rows of df"
	# print df.head

# print "method 2"
# print next(os.walk(args.in_folder))[1]
# print next(os.walk(args.in_folder))[0]
# print "first print base path"
print_paths_and_log_to_csv(next(os.walk(args.in_folder))[0])
# print "print paths"
# for i in next(os.walk(args.in_folder))[1]:
# 	print os.path.join(next(os.walk(args.in_folder))[0],i)
# 	print "contents"
# 	#FUNCTIONIZE THIS + saving to pandas
# 	print_paths_and_log_to_csv(os.path.join(next(os.walk(args.in_folder))[0],i))
	# j=os.path.join(next(os.walk(args.in_folder))[0],i)
	# print next(os.walk(j))[1]

		# a.write(str(f)+'\n')

