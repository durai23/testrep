
module load GCCcore/.5.3.0
module load FreeSurfer/5.3.0-centis4_x84_64
source $FREESURFER_HOME/SetUpFreeSurfer.sh
setenv SUBJECTS_DIR $WORK/Data_T1
cd $SUBJECTS_DIR

srun --partition=batch --time=20:00:00 --cpus-per-task=1 --ntasks-per-node=2 --nodes=1 --mail-type=END --mail-user=c.jockwitz@fz-juelich.de --job-name=freesurfer --ntasks=2 --cpu_bind=none recon-all -subject 998331_para_1 -i 998331_V1/T1.nii -all -time & recon-all -subject 998331_para_2 -i 998331_V1/T1.nii -all -time
