darasan@imecas09:/data/Team_Caspers/JURECA$ cat recon_all_T1_execute.sh 
#load FreeSurfer
module load FreeSurfer/5.3.0-centos6_x86_64
export SUBJECTS_DIR=$WORK/FS/reconall
source $FREESURFER_HOME/SetUpFreeSurfer.sh

#load FSL
module use /usr/local/software/jureca/OtherStages
module load Stages/Current  GCC/5.4.0 FSL/5.0.9
source /usr/local/software/jureca/Stages/2016b/software/FSL/5.0.9-GCC-5.4.0/fsl/etc/fslconf/fsl.sh

#run splited jobs
for i in recona?; do
srun --partition=batch --time=23:59:00 --cpus-per-task=48 --ntasks-per-node=1 --nodes=1 --mail-type=END --job-name=fs_recon_$i --ntasks=1 --cpu_bind=none $HOME/fzj_parallel.sh -j 24 -m $i &
done

