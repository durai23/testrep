#template program for printing full path of all files in
# input folder
# only input is manual
# make it argparse - refer 2_data_paths.py
# then pass all fixed drive letters

import os

a=open(r'C:\Users\synapse\Desktop\fzj_1.txt','w')
for path, subdirs, files in os.walk(r'N:\fzj_temp_dump'):
	for filename in files:
		f=os.path.join(path,filename)
		a.write(str(f)+'\n')