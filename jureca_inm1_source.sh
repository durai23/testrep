#!/bin/bash
:'
-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2018                           EasyBuild/3.5.1                    (D)    LLVM/4.0.1                     (D)    Vampir/9.3.0                    ispc/1.9.1
   AllineaForge/7.1                       FreeSurfer/6.0.0                          MC/4.8.19                             VirtualGL/default               itac/2018.0.015
   AllineaPerformanceReports/7.1          GC3Pie/2.4.2                              Mercurial/4.0.1-Python-2.7.14         X11/20170314                    jemalloc/5.0.1
   Atom/1.21.1                            GDB/8.0.1                                 NCCL/2.1.4-CUDA-9.0.176               Xerces-C++/3.2.0                meld/3.16.4-Python-2.7.14
   Autotools/20150215                     GIMP/2.9.4                                OpenEXR/2.2.0                         Xpra/1.0.8-Python-2.7.14        memkind/1.6.0
   Bazel/0.7.0                            GMP/6.1.2                                 PAPI/5.5.1                            cURL/7.56.0                     numactl/2.0.11
   Blender/2.79-binary                    GPicView/0.2.5                            Perl/5.26.1                           cppcheck/1.81                   p7zip/16.02
   CFITSIO/3.420                          Grace/5.1.25                              Polygon2/2.0.8-Python-2.7.14          cuDNN/6.0-CUDA-8.0.61           tbb/2018.0.128
   CMake/3.9.4                            Graphviz/2.40.1                           PostgreSQL/9.6.5                      cuDNN/7.0.3-CUDA-9.0.176 (D)    tbb/2018.1.163            (D)
   CMake/3.10.2                  (D)      HDFView/2.14-Java-1.8.0_144               PyGTK/2.24.0-Python-2.7.14            flex/2.6.4                      tcsh/6.20.00
   CUDA/8.0.61                   (g)      ICA-AROMA/0.4.3-beta-Python-2.7.14        PyOpenGL/3.1.1a1-Python-2.7.14        git/2.14.2                      tmux/2.6
   CUDA/9.0.176                  (g,D)    IDL/8.5.1                                 PyQt/4.12.1-Python-2.7.14             gnuplot/5.2.0                   unzip/6.0
   CVS/1.11.23                            ImageMagick/6.9.9-19                      Python/2.7.14                         imake/1.0.7                     wgrib/1.8.1.2c
   Camino/20161122                        Inspector/2018                            Python/3.6.3                   (D)    intel-para/2017b-mt             xdiskusage/1.51
   Cube/4.3.5                             JUBE/2.1.4                                Subversion/1.9.7                      intel-para/2017b         (D)    zsh/5.4.2
   Doxygen/1.8.13                         JUBE/2.2.0                         (D)    TotalView/2017.2.11                   intel-para/2017b.1-mt
   EasyBuild/3.4.0                        Java/1.8.0_144                            TotalView/2017.3.8             (D)    intel-para/2017b.1
   EasyBuild/3.5.0                        LLVM/3.9.1                                VTune/2018                            ipp/2018.0.128

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0    GCC/7.2.0 (D)    Intel/2017.5.239-GCC-5.4.0    Intel/2018.0.128-GCC-5.4.0 (D)    Intel/2018.1.163-GCC-5.4.0    PGI/17.9-GCC-5.4.0

---------------------------------------------------------------------------------------- Recommended defaults -----------------------------------------------------------------------------------------
   defaults/CPU

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)

-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2017b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2017b/UI/Compilers:/usr/local/software/jureca/Stages/2017b/UI/Tools:/usr/local/software/jureca/Stages/2017b/UI/Defaults:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/software/jureca/Stages/2017b/software/binutils/2.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2017b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module purge --force"
echo $EXE
eval "$EXE"
:'
---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0    GCC/7.2.0 (D)    Intel/2017.5.239-GCC-5.4.0    Intel/2018.0.128-GCC-5.4.0 (D)    Intel/2018.1.163-GCC-5.4.0    PGI/17.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2018                           Camino/20161122        HDFView/2.14-Java-1.8.0_144        NCCL/2.1.4-CUDA-9.0.176        cuDNN/6.0-CUDA-8.0.61           intel-para/2017b.1    tbb/2018.1.163 (D)
   AllineaForge/7.1                       EasyBuild/3.4.0        Inspector/2018                     TotalView/2017.2.11            cuDNN/7.0.3-CUDA-9.0.176 (D)    ipp/2018.0.128
   AllineaPerformanceReports/7.1          EasyBuild/3.5.0        JUBE/2.1.4                         TotalView/2017.3.8      (D)    intel-para/2017b-mt             ispc/1.9.1
   CUDA/8.0.61                   (g)      EasyBuild/3.5.1 (D)    JUBE/2.2.0                  (D)    VTune/2018                     intel-para/2017b         (D)    itac/2018.0.015
   CUDA/9.0.176                  (g,D)    GC3Pie/2.4.2           Java/1.8.0_144                     Vampir/9.3.0                   intel-para/2017b.1-mt           tbb/2018.0.128

---------------------------------------------------------------------------------------- Recommended defaults -----------------------------------------------------------------------------------------
   defaults/CPU

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2017b/UI/Compilers:/usr/local/software/jureca/Stages/2017b/UI/Tools:/usr/local/software/jureca/Stages/2017b/UI/Defaults:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module use /usr/local/software/jureca/OtherStages"
#ONLY USE OF THIS COMMAND IS TO ADD Other Stages IN TURN IN ORDER TO LOAD Stages/2017.lua symlinked as 2016b.lua
echo $EXE
eval "$EXE"
:'
-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)    Stages/2018a (S,D)

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0    GCC/7.2.0 (D)    Intel/2017.5.239-GCC-5.4.0    Intel/2018.0.128-GCC-5.4.0 (D)    Intel/2018.1.163-GCC-5.4.0    PGI/17.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2018                           Camino/20161122        HDFView/2.14-Java-1.8.0_144        NCCL/2.1.4-CUDA-9.0.176        cuDNN/6.0-CUDA-8.0.61           intel-para/2017b.1    tbb/2018.1.163 (D)
   AllineaForge/7.1                       EasyBuild/3.4.0        Inspector/2018                     TotalView/2017.2.11            cuDNN/7.0.3-CUDA-9.0.176 (D)    ipp/2018.0.128
   AllineaPerformanceReports/7.1          EasyBuild/3.5.0        JUBE/2.1.4                         TotalView/2017.3.8      (D)    intel-para/2017b-mt             ispc/1.9.1
   CUDA/8.0.61                   (g)      EasyBuild/3.5.1 (D)    JUBE/2.2.0                  (D)    VTune/2018                     intel-para/2017b         (D)    itac/2018.0.015
   CUDA/9.0.176                  (g,D)    GC3Pie/2.4.2           Java/1.8.0_144                     Vampir/9.3.0                   intel-para/2017b.1-mt           tbb/2018.0.128

---------------------------------------------------------------------------------------- Recommended defaults -----------------------------------------------------------------------------------------
   defaults/CPU

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/OtherStages:/usr/local/software/jureca/Stages/2017b/UI/Compilers:/usr/local/software/jureca/Stages/2017b/UI/Tools:/usr/local/software/jureca/Stages/2017b/UI/Defaults:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
echo 

EXE="module load Stages/2016b"
#This module will reset your module environment for the requested stage
#We probably dont need the staging system on imecas - you can just replicate it brute force
#ONLY USE OF THIS STEP - SETS MODULEPATH - /usr/local/software/jureca/Stages/2016b/UI/ <Compilers,Defaults,Tools>
#setenv("STAGE", stage)
#setenv("SOFTWAREROOT", softwareroot)
echo $EXE
eval "$EXE"
:'
-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                   CMake/3.6.2            EasyBuild/3.1.0                 (D)    Java/1.8.0_102                    Vampir/9.1.0               itac/2017.1.024
   Advisor/2017_update2            (D)    CUDA/8.0.44     (g)    FreeSurfer/5.3.0-centos6_x86_64        TotalView/2016T.07.11-beta        flex/2.6.0                 tbb/2017.0.098
   AllineaForge/6.1.2                     EasyBuild/2.8.2        GC3Pie/2.4.2                           TotalView/2016.06.21       (D)    intel-para/2016b-mt
   AllineaPerformanceReports/6.1.2        EasyBuild/2.9.0        Inspector/2017_update1                 VTune/2017_update1                intel-para/2016b    (D)
   Blender/2.78-binary                    EasyBuild/3.0.1        JUBE/2.1.3                             VTune/2017                 (D)    ipp/2017.1.132

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module load GCC/5.4.0"
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10    GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1    Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9       GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                   f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                   Cube/4.3.4                             Graphviz/2.38.0                           PostgreSQL/9.6.0                           Xerces-C++/3.1.4           itac/2017.1.024
   Advisor/2017_update2            (D)    Doxygen/1.8.12                         IDL/8.5.1                                 PyOpenGL/3.1.1a1-Python-2.7.12-bare        cURL/7.50.3                meld/3.16.3
   AllineaForge/6.1.2                     EasyBuild/2.8.2                        Inspector/2017_update1                    Subversion/1.9.4                           cppcheck/1.76.1            p7zip/16.02
   AllineaPerformanceReports/6.1.2        EasyBuild/2.9.0                        JUBE/2.1.3                                TotalView/2016T.07.11-beta                 flex/2.6.0                 tbb/2017.0.098
   Autotools/20150215                     EasyBuild/3.0.1                        Java/1.8.0_102                            TotalView/2016.06.21                (D)    flex/2.6.0          (D)    tcsh/6.19.00
   Blender/2.78-binary                    EasyBuild/3.1.0                 (D)    LLVM/3.8.1                                VTune/2017_update1                         git/2.10.0                 tmux/2.3
   CFITSIO/3.39                           FreeSurfer/5.3.0-centos6_x86_64        LLVM/3.9.0                         (D)    VTune/2017                          (D)    gnuplot/5.0.5              wgrib/1.8.1.2c
   CMake/3.6.2                     (D)    GC3Pie/2.4.2                           MC/4.8.18                                 Valgrind/3.11.0                            imake/1.0.7                xdiskusage/1.51
   CMake/3.6.2                            GDB/7.11.1                             Mercurial/3.9.2-Python-2.7.12-bare        Valgrind/3.12.0                     (D)    intel-para/2016b-mt        zsh/5.2
   CUDA/8.0.44                     (g)    GPicView/0.2.5                         PAPI/5.5.0                                Vampir/9.1.0                               intel-para/2016b    (D)
   CVS/1.11.23                            Grace/5.1.25                           Perl/5.24.0                               X.Org/2016-09-22                           ipp/2017.1.132

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module load Python/2.7.12"
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10    GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9       GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                   EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)    EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                     EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0  (D)    ipp/2017.1.132
   AllineaPerformanceReports/6.1.2        EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                     FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                    GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                           GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)    GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                            Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                            IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                             Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                         JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module load FSL/5.0.9"
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0  (D)    ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="source ${FSLDIR}/etc/fslconf/fsl.sh"
#THIS COMMAND SETS FSL ENV VARS - NO HCANGES TO PATH OR MODULEPATH
#SOURCES FSL INIT FILES
#PAINFULLY SEE IF THIS VARS ARE SET ON IMECAS ALREADY OR NEED TO BE SET MANUALY
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0  (D)    ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
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
:'
----------------------------------------------------------------------- Packages compiled with ParaStationMPI and GCC compilers -----------------------------------------------------------------------
   ABINIT/8.0.8b                     HDF5/1.8.17                        ParaView/5.1.2                               (D)    TAU/2.25.2                        netCDF-Fortran/4.4.4
   ARPACK-NG/3.4.0                   Hypre/2.11.1-bigint                QuantumESPRESSO/6.0                                 Valgrind/3.12.0            (D)    netCDF/4.4.1
   Boost/1.61.0-Python-2.7.12        Hypre/2.11.1                (D)    R/3.3.1                                             VampirServer/9.1.0                netcdf4-python/1.2.4-Python-2.7.12
   CGAL/4.9-Python-2.7.12            IOR/3.0.1-mpiio                    RELION/1.4                                          buildenv/gpsolf                   netcdf4-python/1.2.4-Python-3.5.2  (D)
   COMPSs/1.4                        LinkTest/1.2p1                     SCOTCH/6.0.4                                        darshan-runtime/3.1.1             numba/0.28.1-Python-2.7.12
   CP2K/4.1-plumed-elpa              MUMPS/5.0.2                        SIONlib/1.6.2                                       darshan-runtime/3.1.2      (D)    numba/0.28.1-Python-3.5.2          (D)
   ELPA/2016.05.003-hybrid           MUST/1.5.0-Python-2.7.12           ScaLAPACK/2.0.2-OpenBLAS-0.2.19-LAPACK-3.6.1        darshan-util/3.1.1                parallel-netcdf/1.7.0
   ELPA/2016.05.003-pure-mpi  (D)    NCO/4.6.1                          Scalasca/2.3.1                                      darshan-util/3.1.2         (D)    sprng/1
   Elemental/0.85                    Octave/4.0.3                       SciPy-Stack/2016b-Python-2.7.12                     h5py/2.6.0-Python-2.7.12          sprng/5                            (D)
   Extrae/3.3.0                      OpenCV/2.4.13-Python-2.7.12        SciPy-Stack/2016b-Python-3.5.2               (D)    mpi4py/2.0.0-Python-2.7.12 (L)    sundials/2.7.0
   FFTW/3.3.5                        PLUMED/2.2.3                       Score-P/3.0-p1                                      mpi4py/2.0.0-Python-3.5.2  (D)
   GSL/2.2.1                         ParMETIS/4.0.3                     Silo/4.10.2                                         ncview/2.1.7
   HDF/4.2.12                        ParaView/5.1.2-OSMesa              SuiteSparse/4.5.3-METIS-5.1.0                       netCDF-C++4/4.3.0

------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1 (L)

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0         ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/pscom/Default/bin:/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
echo

EXE="module use /data/inm1/mapping/software/2016b/modules"
#THIS COMMANDS PURELY ADDS /data/inm1/mapping/software/2016b/modules TO MODULEPATH - NO OTHER CHANGES
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------------ /data/inm1/mapping/software/2016b/modules ------------------------------------------------------------------------------------
   ANTs/2.1.0          FIX/1.065              FZJ_dMRI/1.0.0        MRtrix/0.3.15             NeuroImaging_Python3/1.0.0        Snakemake/3.10.2              Snakemake_extra/3.13.3 (D)
   ANTs_extra/2.1.0    FSL/5.0.10      (D)    MDT/0.9.30            MRtrix/3.0_RC1_189        NeuroImaging_Python3/1.1.0 (D)    Snakemake/3.11.2              brainvisa/4.4.0
   AROMA/0.3beta       FSL_extra/5.0.9        MDT/0.9.31            MRtrix/3.0_RC3     (D)    Octave/3.8.2                      Snakemake/3.13.3       (D)
   CAT/0.8             FZJ/1.0.0              MDT/0.10.5     (D)    Nano/2.7.4                Parallel/2016.12.22               Snakemake_extra/3.11.2

----------------------------------------------------------------------- Packages compiled with ParaStationMPI and GCC compilers -----------------------------------------------------------------------
   ABINIT/8.0.8b                     HDF5/1.8.17                        ParaView/5.1.2                               (D)    TAU/2.25.2                        netCDF-Fortran/4.4.4
   ARPACK-NG/3.4.0                   Hypre/2.11.1-bigint                QuantumESPRESSO/6.0                                 Valgrind/3.12.0            (D)    netCDF/4.4.1
   Boost/1.61.0-Python-2.7.12        Hypre/2.11.1                (D)    R/3.3.1                                             VampirServer/9.1.0                netcdf4-python/1.2.4-Python-2.7.12
   CGAL/4.9-Python-2.7.12            IOR/3.0.1-mpiio                    RELION/1.4                                          buildenv/gpsolf                   netcdf4-python/1.2.4-Python-3.5.2  (D)
   COMPSs/1.4                        LinkTest/1.2p1                     SCOTCH/6.0.4                                        darshan-runtime/3.1.1             numba/0.28.1-Python-2.7.12
   CP2K/4.1-plumed-elpa              MUMPS/5.0.2                        SIONlib/1.6.2                                       darshan-runtime/3.1.2      (D)    numba/0.28.1-Python-3.5.2          (D)
   ELPA/2016.05.003-hybrid           MUST/1.5.0-Python-2.7.12           ScaLAPACK/2.0.2-OpenBLAS-0.2.19-LAPACK-3.6.1        darshan-util/3.1.1                parallel-netcdf/1.7.0
   ELPA/2016.05.003-pure-mpi  (D)    NCO/4.6.1                          Scalasca/2.3.1                                      darshan-util/3.1.2         (D)    sprng/1
   Elemental/0.85                    Octave/4.0.3                (D)    SciPy-Stack/2016b-Python-2.7.12                     h5py/2.6.0-Python-2.7.12          sprng/5                            (D)
   Extrae/3.3.0                      OpenCV/2.4.13-Python-2.7.12        SciPy-Stack/2016b-Python-3.5.2               (D)    mpi4py/2.0.0-Python-2.7.12 (L)    sundials/2.7.0
   FFTW/3.3.5                        PLUMED/2.2.3                       Score-P/3.0-p1                                      mpi4py/2.0.0-Python-3.5.2  (D)
   GSL/2.2.1                         ParMETIS/4.0.3                     Silo/4.10.2                                         ncview/2.1.7
   HDF/4.2.12                        ParaView/5.1.2-OSMesa              SuiteSparse/4.5.3-METIS-5.1.0                       netCDF-C++4/4.3.0

------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1 (L)

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0         ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/data/inm1/mapping/software/2016b/modules:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/pscom/Default/bin:/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
echo

EXE="module load FSL_extra/5.0.9"
#this does not change the MODULEPATH
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------------ /data/inm1/mapping/software/2016b/modules ------------------------------------------------------------------------------------
   ANTs/2.1.0          FIX/1.065              FZJ_dMRI/1.0.0        MRtrix/0.3.15             NeuroImaging_Python3/1.0.0        Snakemake/3.10.2              Snakemake_extra/3.13.3 (D)
   ANTs_extra/2.1.0    FSL/5.0.10      (D)    MDT/0.9.30            MRtrix/3.0_RC1_189        NeuroImaging_Python3/1.1.0 (D)    Snakemake/3.11.2              brainvisa/4.4.0
   AROMA/0.3beta       FSL_extra/5.0.9 (L)    MDT/0.9.31            MRtrix/3.0_RC3     (D)    Octave/3.8.2                      Snakemake/3.13.3       (D)
   CAT/0.8             FZJ/1.0.0              MDT/0.10.5     (D)    Nano/2.7.4                Parallel/2016.12.22               Snakemake_extra/3.11.2

----------------------------------------------------------------------- Packages compiled with ParaStationMPI and GCC compilers -----------------------------------------------------------------------
   ABINIT/8.0.8b                     HDF5/1.8.17                        ParaView/5.1.2                               (D)    TAU/2.25.2                        netCDF-Fortran/4.4.4
   ARPACK-NG/3.4.0                   Hypre/2.11.1-bigint                QuantumESPRESSO/6.0                                 Valgrind/3.12.0            (D)    netCDF/4.4.1
   Boost/1.61.0-Python-2.7.12        Hypre/2.11.1                (D)    R/3.3.1                                             VampirServer/9.1.0                netcdf4-python/1.2.4-Python-2.7.12
   CGAL/4.9-Python-2.7.12            IOR/3.0.1-mpiio                    RELION/1.4                                          buildenv/gpsolf                   netcdf4-python/1.2.4-Python-3.5.2  (D)
   COMPSs/1.4                        LinkTest/1.2p1                     SCOTCH/6.0.4                                        darshan-runtime/3.1.1             numba/0.28.1-Python-2.7.12
   CP2K/4.1-plumed-elpa              MUMPS/5.0.2                        SIONlib/1.6.2                                       darshan-runtime/3.1.2      (D)    numba/0.28.1-Python-3.5.2          (D)
   ELPA/2016.05.003-hybrid           MUST/1.5.0-Python-2.7.12           ScaLAPACK/2.0.2-OpenBLAS-0.2.19-LAPACK-3.6.1        darshan-util/3.1.1                parallel-netcdf/1.7.0
   ELPA/2016.05.003-pure-mpi  (D)    NCO/4.6.1                          Scalasca/2.3.1                                      darshan-util/3.1.2         (D)    sprng/1
   Elemental/0.85                    Octave/4.0.3                (D)    SciPy-Stack/2016b-Python-2.7.12                     h5py/2.6.0-Python-2.7.12          sprng/5                            (D)
   Extrae/3.3.0                      OpenCV/2.4.13-Python-2.7.12        SciPy-Stack/2016b-Python-3.5.2               (D)    mpi4py/2.0.0-Python-2.7.12 (L)    sundials/2.7.0
   FFTW/3.3.5                        PLUMED/2.2.3                       Score-P/3.0-p1                                      mpi4py/2.0.0-Python-3.5.2  (D)
   GSL/2.2.1                         ParMETIS/4.0.3                     Silo/4.10.2                                         ncview/2.1.7
   HDF/4.2.12                        ParaView/5.1.2-OSMesa              SuiteSparse/4.5.3-METIS-5.1.0                       netCDF-C++4/4.3.0

------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1 (L)

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0         ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/data/inm1/mapping/software/2016b/modules:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/data/inm1/mapping/software/2016b/installed/FSL_extra-5.0.9/bin:/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/pscom/Default/bin:/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module load MRtrix/0.3.15"
#this does not change the MODULEPATH
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------------ /data/inm1/mapping/software/2016b/modules ------------------------------------------------------------------------------------
   ANTs/2.1.0          FIX/1.065              FZJ_dMRI/1.0.0        MRtrix/0.3.15      (L)    NeuroImaging_Python3/1.0.0        Snakemake/3.10.2              Snakemake_extra/3.13.3 (D)
   ANTs_extra/2.1.0    FSL/5.0.10      (D)    MDT/0.9.30            MRtrix/3.0_RC1_189        NeuroImaging_Python3/1.1.0 (D)    Snakemake/3.11.2              brainvisa/4.4.0
   AROMA/0.3beta       FSL_extra/5.0.9 (L)    MDT/0.9.31            MRtrix/3.0_RC3     (D)    Octave/3.8.2                      Snakemake/3.13.3       (D)
   CAT/0.8             FZJ/1.0.0              MDT/0.10.5     (D)    Nano/2.7.4                Parallel/2016.12.22               Snakemake_extra/3.11.2

----------------------------------------------------------------------- Packages compiled with ParaStationMPI and GCC compilers -----------------------------------------------------------------------
   ABINIT/8.0.8b                     HDF5/1.8.17                        ParaView/5.1.2                               (D)    TAU/2.25.2                        netCDF-Fortran/4.4.4
   ARPACK-NG/3.4.0                   Hypre/2.11.1-bigint                QuantumESPRESSO/6.0                                 Valgrind/3.12.0            (D)    netCDF/4.4.1
   Boost/1.61.0-Python-2.7.12        Hypre/2.11.1                (D)    R/3.3.1                                             VampirServer/9.1.0                netcdf4-python/1.2.4-Python-2.7.12
   CGAL/4.9-Python-2.7.12            IOR/3.0.1-mpiio                    RELION/1.4                                          buildenv/gpsolf                   netcdf4-python/1.2.4-Python-3.5.2  (D)
   COMPSs/1.4                        LinkTest/1.2p1                     SCOTCH/6.0.4                                        darshan-runtime/3.1.1             numba/0.28.1-Python-2.7.12
   CP2K/4.1-plumed-elpa              MUMPS/5.0.2                        SIONlib/1.6.2                                       darshan-runtime/3.1.2      (D)    numba/0.28.1-Python-3.5.2          (D)
   ELPA/2016.05.003-hybrid           MUST/1.5.0-Python-2.7.12           ScaLAPACK/2.0.2-OpenBLAS-0.2.19-LAPACK-3.6.1        darshan-util/3.1.1                parallel-netcdf/1.7.0
   ELPA/2016.05.003-pure-mpi  (D)    NCO/4.6.1                          Scalasca/2.3.1                                      darshan-util/3.1.2         (D)    sprng/1
   Elemental/0.85                    Octave/4.0.3                (D)    SciPy-Stack/2016b-Python-2.7.12                     h5py/2.6.0-Python-2.7.12          sprng/5                            (D)
   Extrae/3.3.0                      OpenCV/2.4.13-Python-2.7.12        SciPy-Stack/2016b-Python-3.5.2               (D)    mpi4py/2.0.0-Python-2.7.12 (L)    sundials/2.7.0
   FFTW/3.3.5                        PLUMED/2.2.3                       Score-P/3.0-p1                                      mpi4py/2.0.0-Python-3.5.2  (D)
   GSL/2.2.1                         ParMETIS/4.0.3                     Silo/4.10.2                                         ncview/2.1.7
   HDF/4.2.12                        ParaView/5.1.2-OSMesa              SuiteSparse/4.5.3-METIS-5.1.0                       netCDF-C++4/4.3.0

------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1 (L)

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0         ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/data/inm1/mapping/software/2016b/modules:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/scripts:/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/bin:/data/inm1/mapping/software/2016b/installed/FSL_extra-5.0.9/bin:/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/pscom/Default/bin:/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module load ANTs/2.1.0"
#this changes the MODULEPATH BUT ONLY REARRANGES
echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1 (L)

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0  (D)    ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

------------------------------------------------------------------------------------ /data/inm1/mapping/software/2016b/modules ------------------------------------------------------------------------------------
   ANTs/2.1.0       (L)    FIX/1.065              FZJ_dMRI/1.0.0        MRtrix/0.3.15      (L)    NeuroImaging_Python3/1.0.0        Snakemake/3.10.2              Snakemake_extra/3.13.3 (D)
   ANTs_extra/2.1.0        FSL/5.0.10      (D)    MDT/0.9.30            MRtrix/3.0_RC1_189        NeuroImaging_Python3/1.1.0 (D)    Snakemake/3.11.2              brainvisa/4.4.0
   AROMA/0.3beta           FSL_extra/5.0.9 (L)    MDT/0.9.31            MRtrix/3.0_RC3     (D)    Octave/3.8.2                      Snakemake/3.13.3       (D)
   CAT/0.8                 FZJ/1.0.0              MDT/0.10.5     (D)    Nano/2.7.4                Parallel/2016.12.22               Snakemake_extra/3.11.2

----------------------------------------------------------------------- Packages compiled with ParaStationMPI and GCC compilers -----------------------------------------------------------------------
   ABINIT/8.0.8b                     HDF5/1.8.17                        ParaView/5.1.2                               (D)    TAU/2.25.2                        netCDF-Fortran/4.4.4
   ARPACK-NG/3.4.0                   Hypre/2.11.1-bigint                QuantumESPRESSO/6.0                                 Valgrind/3.12.0                   netCDF/4.4.1
   Boost/1.61.0-Python-2.7.12        Hypre/2.11.1                (D)    R/3.3.1                                             VampirServer/9.1.0                netcdf4-python/1.2.4-Python-2.7.12
   CGAL/4.9-Python-2.7.12            IOR/3.0.1-mpiio                    RELION/1.4                                          buildenv/gpsolf                   netcdf4-python/1.2.4-Python-3.5.2  (D)
   COMPSs/1.4                        LinkTest/1.2p1                     SCOTCH/6.0.4                                        darshan-runtime/3.1.1             numba/0.28.1-Python-2.7.12
   CP2K/4.1-plumed-elpa              MUMPS/5.0.2                        SIONlib/1.6.2                                       darshan-runtime/3.1.2      (D)    numba/0.28.1-Python-3.5.2          (D)
   ELPA/2016.05.003-hybrid           MUST/1.5.0-Python-2.7.12           ScaLAPACK/2.0.2-OpenBLAS-0.2.19-LAPACK-3.6.1        darshan-util/3.1.1                parallel-netcdf/1.7.0
   ELPA/2016.05.003-pure-mpi  (D)    NCO/4.6.1                          Scalasca/2.3.1                                      darshan-util/3.1.2         (D)    sprng/1
   Elemental/0.85                    Octave/4.0.3                (D)    SciPy-Stack/2016b-Python-2.7.12                     h5py/2.6.0-Python-2.7.12          sprng/5                            (D)
   Extrae/3.3.0                      OpenCV/2.4.13-Python-2.7.12        SciPy-Stack/2016b-Python-3.5.2               (D)    mpi4py/2.0.0-Python-2.7.12 (L)    sundials/2.7.0
   FFTW/3.3.5                        PLUMED/2.2.3                       Score-P/3.0-p1                                      mpi4py/2.0.0-Python-3.5.2  (D)
   GSL/2.2.1                         ParMETIS/4.0.3                     Silo/4.10.2                                         ncview/2.1.7
   HDF/4.2.12                        ParaView/5.1.2-OSMesa              SuiteSparse/4.5.3-METIS-5.1.0                       netCDF-C++4/4.3.0

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/data/inm1/mapping/software/2016b/modules:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/data/inm1/mapping/software/2016b/installed/ANTs-2.1.0/Scripts:/data/inm1/mapping/software/2016b/installed/ANTs-2.1.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/scripts:/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/bin:/data/inm1/mapping/software/2016b/installed/FSL_extra-5.0.9/bin:/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/pscom/Default/bin:/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module load ANTs_extra/2.1.0"
#THIS DOES NOT CHANGE MDULEPATH
#prereq("ANTs/2.1.0")

echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1 (L)

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0  (D)    ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

------------------------------------------------------------------------------------ /data/inm1/mapping/software/2016b/modules ------------------------------------------------------------------------------------
   ANTs/2.1.0       (L)    FIX/1.065              FZJ_dMRI/1.0.0        MRtrix/0.3.15      (L)    NeuroImaging_Python3/1.0.0        Snakemake/3.10.2              Snakemake_extra/3.13.3 (D)
   ANTs_extra/2.1.0 (L)    FSL/5.0.10      (D)    MDT/0.9.30            MRtrix/3.0_RC1_189        NeuroImaging_Python3/1.1.0 (D)    Snakemake/3.11.2              brainvisa/4.4.0
   AROMA/0.3beta           FSL_extra/5.0.9 (L)    MDT/0.9.31            MRtrix/3.0_RC3     (D)    Octave/3.8.2                      Snakemake/3.13.3       (D)
   CAT/0.8                 FZJ/1.0.0              MDT/0.10.5     (D)    Nano/2.7.4                Parallel/2016.12.22               Snakemake_extra/3.11.2

----------------------------------------------------------------------- Packages compiled with ParaStationMPI and GCC compilers -----------------------------------------------------------------------
   ABINIT/8.0.8b                     HDF5/1.8.17                        ParaView/5.1.2                               (D)    TAU/2.25.2                        netCDF-Fortran/4.4.4
   ARPACK-NG/3.4.0                   Hypre/2.11.1-bigint                QuantumESPRESSO/6.0                                 Valgrind/3.12.0                   netCDF/4.4.1
   Boost/1.61.0-Python-2.7.12        Hypre/2.11.1                (D)    R/3.3.1                                             VampirServer/9.1.0                netcdf4-python/1.2.4-Python-2.7.12
   CGAL/4.9-Python-2.7.12            IOR/3.0.1-mpiio                    RELION/1.4                                          buildenv/gpsolf                   netcdf4-python/1.2.4-Python-3.5.2  (D)
   COMPSs/1.4                        LinkTest/1.2p1                     SCOTCH/6.0.4                                        darshan-runtime/3.1.1             numba/0.28.1-Python-2.7.12
   CP2K/4.1-plumed-elpa              MUMPS/5.0.2                        SIONlib/1.6.2                                       darshan-runtime/3.1.2      (D)    numba/0.28.1-Python-3.5.2          (D)
   ELPA/2016.05.003-hybrid           MUST/1.5.0-Python-2.7.12           ScaLAPACK/2.0.2-OpenBLAS-0.2.19-LAPACK-3.6.1        darshan-util/3.1.1                parallel-netcdf/1.7.0
   ELPA/2016.05.003-pure-mpi  (D)    NCO/4.6.1                          Scalasca/2.3.1                                      darshan-util/3.1.2         (D)    sprng/1
   Elemental/0.85                    Octave/4.0.3                (D)    SciPy-Stack/2016b-Python-2.7.12                     h5py/2.6.0-Python-2.7.12          sprng/5                            (D)
   Extrae/3.3.0                      OpenCV/2.4.13-Python-2.7.12        SciPy-Stack/2016b-Python-3.5.2               (D)    mpi4py/2.0.0-Python-2.7.12 (L)    sundials/2.7.0
   FFTW/3.3.5                        PLUMED/2.2.3                       Score-P/3.0-p1                                      mpi4py/2.0.0-Python-3.5.2  (D)
   GSL/2.2.1                         ParMETIS/4.0.3                     Silo/4.10.2                                         ncview/2.1.7
   HDF/4.2.12                        ParaView/5.1.2-OSMesa              SuiteSparse/4.5.3-METIS-5.1.0                       netCDF-C++4/4.3.0

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/data/inm1/mapping/software/2016b/modules:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/data/inm1/mapping/software/2016b/installed/ANTs_extra-2.1.0/Scripts:/data/inm1/mapping/software/2016b/installed/ANTs-2.1.0/Scripts:/data/inm1/mapping/software/2016b/installed/ANTs-2.1.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/scripts:/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/bin:/data/inm1/mapping/software/2016b/installed/FSL_extra-5.0.9/bin:/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/pscom/Default/bin:/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'
EXE="module load FZJ/1.0.0"
#THIS DOES NOT CHANGE MDULEPATH
#-- prereq("FSL/5.0.9")
#-- prereq("MRtrix/0.3.15")
#prepend_path("PATH", prefix)
#-- prepend_path("PYTHONPATH", prefix)
#-- setenv("FZJDIR", prefix)


echo $EXE
eval "$EXE"
:'
------------------------------------------------------------------------------ MPI runtimes available for GNU compilers -------------------------------------------------------------------------------
   MVAPICH2/2.2-GDR (g)    ParaStationMPI/5.1.5-1 (L)

-------------------------------------------------------------------------------- Packages compiled with GNU compilers ---------------------------------------------------------------------------------
   Eigen/3.2.10        GEOS/3.5.0-Python-2.7.12    Libxc/2.2.2    MPFR/3.1.5                     OpenBLAS/0.2.19-LAPACK-3.6.1        Python/3.5.2  (D)    librsb/1.2.0-rc5
   FSL/5.0.9    (L)    GMP/6.1.1                   METIS/5.1.0    MRtrix/0.3.15-Python-2.7.12    Python/2.7.12                (L)    f90depend/1.5

-------------------------------------------------------------------------------------------- Core packages --------------------------------------------------------------------------------------------
   Advisor/2017_update1                     EasyBuild/2.8.2                        Java/1.8.0_102                      (L)      VTune/2017       (D)    intel-para/2016b-mt
   Advisor/2017_update2            (D)      EasyBuild/2.9.0                        LLVM/3.8.1                                   Valgrind/3.11.0         intel-para/2016b    (D)
   AllineaForge/6.1.2                       EasyBuild/3.0.1                        LLVM/3.9.0                          (L,D)    Valgrind/3.12.0  (D)    ipp/2017.1.132
   AllineaPerformanceReports/6.1.2          EasyBuild/3.1.0                 (D)    MC/4.8.18                                    Vampir/9.1.0            itac/2017.1.024
   Autotools/20150215                       FreeSurfer/5.3.0-centos6_x86_64        Mercurial/3.9.2-Python-2.7.12-bare           X.Org/2016-09-22 (L)    meld/3.16.3
   Blender/2.78-binary                      GC3Pie/2.4.2                           PAPI/5.5.0                                   Xerces-C++/3.1.4        p7zip/16.02
   CFITSIO/3.39                             GDB/7.11.1                             Perl/5.24.0                                  cURL/7.50.3             tbb/2017.0.098
   CMake/3.6.2                     (D)      GPicView/0.2.5                         PostgreSQL/9.6.0                    (L)      cppcheck/1.76.1         tcsh/6.19.00
   CMake/3.6.2                              Grace/5.1.25                           PyOpenGL/3.1.1a1-Python-2.7.12-bare          flex/2.6.0              tmux/2.3
   CUDA/8.0.44                     (g,L)    Graphviz/2.38.0                        Subversion/1.9.4                             flex/2.6.0       (D)    wgrib/1.8.1.2c
   CVS/1.11.23                              IDL/8.5.1                              TotalView/2016T.07.11-beta                   git/2.10.0              xdiskusage/1.51
   Cube/4.3.4                               Inspector/2017_update1                 TotalView/2016.06.21                (D)      gnuplot/5.0.5           zsh/5.2
   Doxygen/1.8.12                           JUBE/2.1.3                             VTune/2017_update1                           imake/1.0.7

------------------------------------------------------------------------------------ /data/inm1/mapping/software/2016b/modules ------------------------------------------------------------------------------------
   ANTs/2.1.0       (L)    FIX/1.065              FZJ_dMRI/1.0.0        MRtrix/0.3.15      (L)    NeuroImaging_Python3/1.0.0        Snakemake/3.10.2              Snakemake_extra/3.13.3 (D)
   ANTs_extra/2.1.0 (L)    FSL/5.0.10      (D)    MDT/0.9.30            MRtrix/3.0_RC1_189        NeuroImaging_Python3/1.1.0 (D)    Snakemake/3.11.2              brainvisa/4.4.0
   AROMA/0.3beta           FSL_extra/5.0.9 (L)    MDT/0.9.31            MRtrix/3.0_RC3     (D)    Octave/3.8.2                      Snakemake/3.13.3       (D)
   CAT/0.8                 FZJ/1.0.0       (L)    MDT/0.10.5     (D)    Nano/2.7.4                Parallel/2016.12.22               Snakemake_extra/3.11.2

----------------------------------------------------------------------- Packages compiled with ParaStationMPI and GCC compilers -----------------------------------------------------------------------
   ABINIT/8.0.8b                     HDF5/1.8.17                        ParaView/5.1.2                               (D)    TAU/2.25.2                        netCDF-Fortran/4.4.4
   ARPACK-NG/3.4.0                   Hypre/2.11.1-bigint                QuantumESPRESSO/6.0                                 Valgrind/3.12.0                   netCDF/4.4.1
   Boost/1.61.0-Python-2.7.12        Hypre/2.11.1                (D)    R/3.3.1                                             VampirServer/9.1.0                netcdf4-python/1.2.4-Python-2.7.12
   CGAL/4.9-Python-2.7.12            IOR/3.0.1-mpiio                    RELION/1.4                                          buildenv/gpsolf                   netcdf4-python/1.2.4-Python-3.5.2  (D)
   COMPSs/1.4                        LinkTest/1.2p1                     SCOTCH/6.0.4                                        darshan-runtime/3.1.1             numba/0.28.1-Python-2.7.12
   CP2K/4.1-plumed-elpa              MUMPS/5.0.2                        SIONlib/1.6.2                                       darshan-runtime/3.1.2      (D)    numba/0.28.1-Python-3.5.2          (D)
   ELPA/2016.05.003-hybrid           MUST/1.5.0-Python-2.7.12           ScaLAPACK/2.0.2-OpenBLAS-0.2.19-LAPACK-3.6.1        darshan-util/3.1.1                parallel-netcdf/1.7.0
   ELPA/2016.05.003-pure-mpi  (D)    NCO/4.6.1                          Scalasca/2.3.1                                      darshan-util/3.1.2         (D)    sprng/1
   Elemental/0.85                    Octave/4.0.3                (D)    SciPy-Stack/2016b-Python-2.7.12                     h5py/2.6.0-Python-2.7.12          sprng/5                            (D)
   Extrae/3.3.0                      OpenCV/2.4.13-Python-2.7.12        SciPy-Stack/2016b-Python-3.5.2               (D)    mpi4py/2.0.0-Python-2.7.12 (L)    sundials/2.7.0
   FFTW/3.3.5                        PLUMED/2.2.3                       Score-P/3.0-p1                                      mpi4py/2.0.0-Python-3.5.2  (D)
   GSL/2.2.1                         ParMETIS/4.0.3                     Silo/4.10.2                                         ncview/2.1.7
   HDF/4.2.12                        ParaView/5.1.2-OSMesa              SuiteSparse/4.5.3-METIS-5.1.0                       netCDF-C++4/4.3.0

---------------------------------------------------------------------------------------------- Compilers ----------------------------------------------------------------------------------------------
   GCC/5.4.0 (L)    Intel/2016.4.258-GCC-5.4.0    Intel/2017.0.098-GCC-5.4.0 (D)    PGI/16.9-GCC-5.4.0

-------------------------------------------------------------------------------------------- Other Stages ---------------------------------------------------------------------------------------------
   Stages/Devel       (S)    Stages/Devel-2017b (S)    Stages/2015b (S)    Stages/2016b (S,L)    Stages/2017b (S)
   Stages/Devel-2017a (S)    Stages/Devel-2018a (S)    Stages/2016a (S)    Stages/2017a (S)      Stages/2018a (S,D)

-------------------------------------------------------------------------------------------- Architectures --------------------------------------------------------------------------------------------
   Architecture/Haswell (S)    Architecture/KNL (S,D)
-bash-4.2$ echo $MODULEPATH
/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCC/5.4.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/GCCcore/5.4.0:/data/inm1/mapping/software/2016b/modules:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/mpi4py/2.0.0-Python-2.7.12:/usr/local/software/jureca/Stages/2016b/modules/all/MPI/GCC/5.4.0/psmpi/5.1.5-1:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/mpi/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/modules/all/Compiler/LLVM/3.9.0:/usr/local/software/jureca/Stages/2016b/UI/Defaults:/usr/local/software/jureca/Stages/2016b/UI/Tools:/usr/local/software/jureca/Stages/2016b/UI/Compilers:/usr/local/software/jureca/OtherStages:/usr/local/software/mod_environment
-bash-4.2$ echo $PATH
/data/inm1/mapping/software/2016b/installed/FZJ-1.0.0:/data/inm1/mapping/software/2016b/installed/ANTs_extra-2.1.0/Scripts:/data/inm1/mapping/software/2016b/installed/ANTs-2.1.0/Scripts:/data/inm1/mapping/software/2016b/installed/ANTs-2.1.0/bin:/usr/local/software/jureca/Stages/2016b/software/binutils/2.27-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/GCCcore/5.4.0/bin:/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/scripts:/data/inm1/mapping/software/2016b/installed/MRtrix-0.3.15/bin:/data/inm1/mapping/software/2016b/installed/FSL_extra-5.0.9/bin:/usr/local/software/jureca/Stages/2016b/software/psmpi/5.1.5-1-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/pscom/Default/bin:/usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/bin:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44:/usr/local/software/jureca/Stages/2016b/software/CUDA/8.0.44/bin:/usr/local/software/jureca/Stages/2016b/software/expat/2.1.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Python/2.7.12-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/PostgreSQL/9.6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102:/usr/local/software/jureca/Stages/2016b/software/Java/1.8.0_102/bin:/usr/local/software/jureca/Stages/2016b/software/libxslt/1.1.29-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libxml2/2.9.4-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/XZ/5.2.2-GCC-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tk/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/X.Org/2016-09-22-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/freetype/2.7-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libpng/1.6.25-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/LLVM/3.9.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/sbin:/usr/local/software/jureca/Stages/2016b/software/eudev/3.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/SQLite/3.14.2-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/Tcl/8.6.6-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/libreadline/7.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/ncurses/6.0-GCCcore-5.4.0/bin:/usr/local/software/jureca/Stages/2016b/software/bzip2/1.0.6-GCC-5.4.0/bin:/usr/local/jsc/bin:/usr/bin:/usr/sbin:/opt/ibutils/bin:/usr/lpp/mmfs/bin
'

#EXE="module load FZJ_dMRI/1.0.0"
#echo $EXE
#eval "$EXE"
