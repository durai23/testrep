#!/bin/bash

EXE="module purge --force"
echo $EXE
eval "$EXE"
#after this you have 2018a stage with UI/{compilers,tools,defaults} AND StdEnv
EXE="module use /usr/local/software/jureca/OtherStages"
echo $EXE
eval "$EXE"
#thsi simply adds Other stages to module avail. Goal is to load 2016b
echo 

EXE="module load Stages/2016b"
#This is the main MODULEPAHT step where a stage is chosen and old stage ie 2018a is removed
#this is done to load GCC 5.4.0 as eixsting GCC is 5.5.0
#
echo $EXE
eval "$EXE"
':
-- Give access to new stage
prepend_path("MODULEPATH", pathJoin(pkgroot, "UI/Compilers"))
prepend_path("MODULEPATH", pathJoin(pkgroot, "UI/Tools"))
prepend_path("MODULEPATH", pathJoin(pkgroot, "UI/Defaults"))
-- Make the module 'sticky' so it is hard to unload
add_property("lmod", "sticky")
-- Mark the module as Booster/KNL ready
--add_property("arch", "sandybridge:knl")
'
EXE="module load GCC/5.4.0"
#done mostly to makes these module availabel - Python/2.7.12 and FSL/5.0.9 and ParaStationMPI/5.1.5-1
echo $EXE
eval "$EXE"
':
local root = "/usr/local/software/jureca/Stages/2016b/software/GCC/5.4.0"
conflict("GCC")
#note two modules loaded below are empty files - so only hcanges to MODULEPATH are the prepend_path commands below
load("GCCcore/.5.4.0")
load("binutils/.2.27")
prepend_path("MODULEPATH", "/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0")
prepend_path("MODULEPATH", "/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0")
setenv("EBROOTGCC", "/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0")
setenv("EBVERSIONGCC", "5.4.0")
setenv("EBDEVELGCC", pathJoin(root, "easybuild/Core-GCC-5.4.0-easybuild-devel"))
family("compiler")
'
EXE="module load Python/2.7.12"
#this does not change MODULEPATH
echo $EXE
eval "$EXE"
':
local root = "/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0"
conflict("Python")
#/usr/local/software/jureca/Stages/2016b/UI/Compilers/GCCcore/.5.4.0.lua 
#/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0
if not isloaded("GCCcore/.5.4.0") then
    load("GCCcore/.5.4.0")
end

if not isloaded("binutils/.2.27") then
    load("binutils/.2.27")
end
#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0/bzip2/.1.0.6.lua
#/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0
if not isloaded("bzip2/.1.0.6") then
    load("bzip2/.1.0.6")
end
#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0/zlib/.1.2.8.lua
#/usr/local/software/jureca/Stages/2016b/software/zlib/1.2.8-GCC-5.4.0
if not isloaded("zlib/.1.2.8") then
    load("zlib/.1.2.8")
end
#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0/libreadline/.7.0.lua
#/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0
if not isloaded("libreadline/.7.0") then
    load("libreadline/.7.0")
end
#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0/ncurses/.6.0.lua
#/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0
if not isloaded("ncurses/.6.0") then
    load("ncurses/.6.0")
end
#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0/SQLite/.3.14.2.lua
#/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0
if not isloaded("SQLite/.3.14.2") then
    load("SQLite/.3.14.2")
end
if not isloaded("Tk/.8.6.6") then
    load("Tk/.8.6.6")
end
if not isloaded("libxml2/.2.9.4") then
    load("libxml2/.2.9.4")
end
#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0/libxslt/.1.1.29.lua
#/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0
if not isloaded("libxslt/.1.1.29") then
    load("libxslt/.1.1.29")
end
if not isloaded("libffi/.3.2.1") then
    load("libffi/.3.2.1")
end
if not isloaded("libyaml/.0.1.7") then
    load("libyaml/.0.1.7")
end
#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0/PostgreSQL/9.6.0.lua
#/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0"
if not isloaded("PostgreSQL/9.6.0") then
    load("PostgreSQL/9.6.0")
end
prepend_path("CPATH", pathJoin(root, "include"))
prepend_path("LD_LIBRARY_PATH", pathJoin(root, "lib"))
prepend_path("LIBRARY_PATH", pathJoin(root, "lib"))
prepend_path("MANPATH", pathJoin(root, "share/man"))
prepend_path("PATH", pathJoin(root, "bin"))
prepend_path("PKG_CONFIG_PATH", pathJoin(root, "lib/pkgconfig"))
setenv("EBROOTPYTHON", root)
setenv("EBVERSIONPYTHON", "2.7.12")
setenv("EBDEVELPYTHON", pathJoin(root, "easybuild/Compiler-GCC-5.4.0-Python-2.7.12-easybuild-devel"))
setenv("EBEXTSLISTPYTHON", "setuptools-28.3.0,pip-8.1.2,nose-1.3.7,blist-1.3.6,paycheck-1.0.2,argparse-1.4.0,pbr-1.10.0,lockfile-0.12.2,Cython-0.24.1,six-1.10.0,dateutil-2.5.3,deap-1.0.2,decorator-4.0.10,arff-2.1.0,pycrypto-2.6.1,ecdsa-0.13,paramiko-2.0.2,pyparsing-2.1.10,netifaces-0.10.5,netaddr-0.7.18,funcsigs-1.0.2,mock-2.0.0,pytz-2016.7,enum34-1.1.6,bitstring-3.1.5,lxml-3.6.4,XlsxWriter-0.9.3,pycparser-2.14,cffi-1.8.3,Pygments-2.1.3,backports.shutil_get_terminal_size-1.0.0,prompt_toolkit-1.0.7,PyYAML-3.12,psycopg2-2.6.2")
'
EXE="module load FSL/5.0.9"
#this does not change MODULEPATH
echo $EXE
eval "$EXE"
':
local root = "/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0"
conflict("FSL")
if not isloaded("GCCcore/.5.4.0") then
    load("GCCcore/.5.4.0")
end
if not isloaded("binutils/.2.27") then
    load("binutils/.2.27")
end

#/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0/freeglut/.3.0.0.lua
#/usr/local/software/jureca/Stages/2016b/software/freeglut/3.0.0-GCCcore-5.4.0
if not isloaded("freeglut/.3.0.0") then
    load("freeglut/.3.0.0")
end
if not isloaded("expat/.2.1.0") then
    load("expat/.2.1.0")
end
if not isloaded("CUDA/8.0.44") then
    load("CUDA/8.0.44")
end
if not isloaded("X.Org/2016-09-22") then
    load("X.Org/2016-09-22")
end
prepend_path("LD_LIBRARY_PATH", pathJoin(root, "fsl/lib"))
prepend_path("PATH", pathJoin(root, "fsl/bin"))
setenv("EBROOTFSL", root)
setenv("EBVERSIONFSL", "5.0.9")
setenv("EBDEVELFSL", pathJoin(root, "easybuild/Compiler-GCC-5.4.0-FSL-5.0.9-easybuild-devel"))
setenv("FSLDIR", "/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl")
'
EXE="source ${FSLDIR}/etc/fslconf/fsl.sh"
echo $EXE
eval "$EXE"
#this does not change MODULEPATH
':
FSLOUTPUTTYPE=NIFTI_GZ
export FSLOUTPUTTYPE
FSLMULTIFILEQUIT=TRUE ; export FSLMULTIFILEQUIT
FSLTCLSH=$FSLDIR/bin/fsltclsh
FSLWISH=$FSLDIR/bin/fslwish
export FSLTCLSH FSLWISH 
FSLLOCKDIR=
FSLMACHINELIST=
FSLREMOTECALL=
export FSLLOCKDIR FSLMACHINELIST FSLREMOTECALL
FSLGECUDAQ=cuda.q
export FSLGECUDAQ
#FSLCONFDIR=$FSLDIR/config
#FSLMACHTYPE=`$FSLDIR/etc/fslconf/fslmachtype.sh`
#export FSLCONFDIR FSLMACHTYPE
if [ -f /usr/local/etc/fslconf/fsl.sh ] ; then
  . /usr/local/etc/fslconf/fsl.sh ;
fi
if [ -f /etc/fslconf/fsl.sh ] ; then
  . /etc/fslconf/fsl.sh ;
fi
if [ -f "${HOME}/.fslconf/fsl.sh" ] ; then
  . "${HOME}/.fslconf/fsl.sh" ;
fi
'
# EXE="module load FreeSurfer/5.3.0-centos6_x86_64"
# echo $EXE
# eval "$EXE"
#EXE="module load Atom/1.14.2"
#echo $EXE
#eval "$EXE"
EXE="module load ParaStationMPI/5.1.5-1 mpi4py/2.0.0-Python-2.7.12"
echo $EXE
eval "$EXE"
#this is done just to load mpi4py/2.0.0-Python-2.7.12
':
local root = "/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0"
conflict("ParaStationMPI")
if not isloaded("GCCcore/.5.4.0") then
    load("GCCcore/.5.4.0")
end
if not isloaded("binutils/.2.27") then
    load("binutils/.2.27")
end
if not isloaded("pscom/.Default") then
    load("pscom/.Default")
end
prepend_path("MODULEPATH", "/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1")
prepend_path("CPATH", pathJoin(root, "include"))
prepend_path("LD_LIBRARY_PATH", pathJoin(root, "lib"))
prepend_path("LIBRARY_PATH", pathJoin(root, "lib"))
prepend_path("MANPATH", pathJoin(root, "share/man"))
prepend_path("PATH", pathJoin(root, "bin"))
prepend_path("PKG_CONFIG_PATH", pathJoin(root, "lib/pkgconfig"))
setenv("EBROOTPSMPI", root)
setenv("EBVERSIONPSMPI", "5.1.5-1")
setenv("EBDEVELPSMPI", pathJoin(root, "easybuild/Compiler-mpi-GCC-5.4.0-ParaStationMPI-5.1.5-1-easybuild-devel"))
family("mpi")
'
#below does nto change the MODULEPATH
':
local root = "/usr/local/software/jureca/Stages/2016b/software/mpi4py/2.0.0-gpsmpi-2016b-Python-2.7.12"
conflict("mpi4py")
if not isloaded("Python/2.7.12") then
    load("Python/2.7.12")
end
prepend_path("MODULEPATH", "/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12")
setenv("EBROOTMPI4PY", root)
setenv("EBVERSIONMPI4PY", "2.0.0")
setenv("EBDEVELMPI4PY", pathJoin(root, "easybuild/MPI-GCC-5.4.0-psmpi-5.1.5-1-mpi4py-2.0.0-Python-2.7.12-easybuild-devel"))
prepend_path("PYTHONPATH", pathJoin(root, "lib/python2.7/site-packages"))
'
echo

EXE="module use /data/inm1/mapping/software/2016b/modules"
echo $EXE
eval "$EXE"

echo

EXE="module load FSL_extra/5.0.9"
echo $EXE
eval "$EXE"
':
prereq("FSL/5.0.9")
local prefix = "/data/inm1/mapping/software/2016b/installed/FSL_extra-5.0.9"
prepend_path("PATH", pathJoin(prefix, "bin"))
prepend_path("LIBRARY_PATH", pathJoin(prefix, "lib"))
prepend_path("LD_LIBRARY_PATH", pathJoin(prefix, "lib"))
'

EXE="module load MRtrix/0.3.15"
echo $EXE
eval "$EXE"
':
prereq("Python/2.7.12")
-- load("GCC/5.4.0")
local prefix = "/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15"
prepend_path("PATH", pathJoin(prefix, "bin"))
prepend_path("PATH", pathJoin(prefix, "scripts"))
prepend_path("LIBRARY_PATH", pathJoin(prefix, "lib"))
prepend_path("LD_LIBRARY_PATH", pathJoin(prefix, "lib"))
setenv("MRTRIXDIR", prefix)
'

EXE="module load ANTs/2.1.0"
echo $EXE
eval "$EXE"
':
-- prereq("GCCcore/.5.4.0")
load("GCC/5.4.0")
local prefix = "/data/inm1/mapping/software/2016b/installed/ANTs-2.1.0"
prepend_path("PATH", pathJoin(prefix, "bin"))
prepend_path("PATH", pathJoin(prefix, "Scripts"))
prepend_path("LIBRARY_PATH", pathJoin(prefix, "lib"))
prepend_path("LD_LIBRARY_PATH", pathJoin(prefix, "lib"))
setenv("ANTSPATH", pathJoin(prefix, "bin/"))
'

EXE="module load ANTs_extra/2.1.0"
echo $EXE
eval "$EXE"
':
prereq("ANTs/2.1.0")
local prefix = "/data/inm1/mapping/software/2016b/installed/ANTs_extra-2.1.0"
prepend_path("PATH", pathJoin(prefix, "Scripts"))
-- prepend_path("LIBRARY_PATH", pathJoin(prefix, "lib"))
-- prepend_path("LD_LIBRARY_PATH", pathJoin(prefix, "lib"))
'

EXE="module load FZJ/1.0.0"
echo $EXE
eval "$EXE"

':
-- prereq("FSL/5.0.9")
-- prereq("MRtrix/0.3.15")
local prefix = "/data/inm1/mapping/software/2016b/installed/FZJ-1.0.0"
prepend_path("PATH", prefix)
-- prepend_path("PYTHONPATH", prefix)
-- setenv("FZJDIR", prefix)
'


#EXE="module load FZJ_dMRI/1.0.0"
#echo $EXE
#eval "$EXE"
