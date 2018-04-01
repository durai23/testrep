# Copyright 2017 Forschungszentrum Juelich

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import os
import numpy as np
from snakemake.io import apply_wildcards
from snakemake import shell
import shutil
from pathlib import Path
import multiprocessing

verbose = True
shell.executable("/bin/bash")

cpu_count = int(multiprocessing.cpu_count() / 2)

#
# source /data/inm1/mapping/snakemake_3.13.3-2016b.source
#
# snakemake -p --nolock -j 48 --cluster-config cluster.json \
# --cluster /data/inm1/mapping/software/2016b/installed/Snakemake_extra-3.13.3/bin/cluster_command.py -- target.nii.gz
#


def is_jureca():
    """
    Checks if this script is running on JURECA or not.
    Returns either 'True' or 'False'
    """
    import os.path
    if not os.path.isfile('/etc/FZJ/systemname'):
        return False
    with open('/etc/FZJ/systemname', 'r') as ff:
        if 'jureca' in ff.read().lower():
            return True
    return False


def is_imecas():
    """
    Checks if this script is running on an imecas server or not.
    Returns either 'True' or 'False'
    """
    import os.path
    if not os.path.isfile('/etc/hostname'):
        return False
    with open('/etc/hostname', 'r') as ff:
        if 'imecas' in ff.read().lower():
            return True
    return False


if is_imecas():
    STORAGE_PATH = '/data/Team_Caspers/Schreiber/JURECA_DATA/RELEASE'
else:
    STORAGE_PATH = '/data/inm1/mapping/RELEASE'


def handle_storage(pattern):
    def handle_wildcards(wildcards):
        f = pattern.format(**wildcards)
        f_data = os.path.join(STORAGE_PATH, f)
        if os.path.exists(f_data):
            return f_data
        return f

    return handle_wildcards


SOURCE_JURECA = """
    source /data/inm1/mapping/software-2016b.source
    module load FZJ_dMRI/1.0.0
    """

SOURCE_IMECAS = """
    mrtrix_prefix="/data/Team_Caspers/Software/MRtrix-0.3.15"
    export PATH="${{mrtrix_prefix}}/bin:${{mrtrix_prefix}}/scripts:${{PATH}}"
    # export LIBRARY_PATH="${{mrtrix_prefix}}/lib:${{LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${{mrtrix_prefix}}/lib:${{LD_LIBRARY_PATH}}"

    ants_prefix="/data/Team_Caspers/Software/ANTs-2.1.0"
    export PATH="${{ants_prefix}}/bin:${{ants_prefix}}/Scripts:${{PATH}}"
    # export LIBRARY_PATH="${{ants_prefix}}/lib:${{LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${{ants_prefix}}/lib:${{LD_LIBRARY_PATH}}"
    export ANTSPATH="${{ants_prefix}}/bin/"

    export PATH="/data/Team_Caspers/Software/FZJ:$PATH"
    export FZJDIR="/data/Team_Caspers/Software/FZJ_dMRI"
    export PATH="/data/Team_Caspers/Software/FZJ_dMRI:$PATH"

    source /data/Team_Caspers/Schreiber/Anaconda/anaconda3/bin/activate
    """

SOURCE_OTHER = SOURCE_IMECAS

if is_jureca():
    SOURCE_COMMAND = SOURCE_JURECA
elif is_imecas():
    SOURCE_COMMAND = SOURCE_IMECAS
else:
    SOURCE_COMMAND = SOURCE_OTHER

onstart:
    if is_jureca():
        from os import getpid
        import subprocess

        print('Initiating jobscript master.')
        pid = getpid()
        cmd = 'python3 /data/inm1/mapping/software/2016b/installed/Snakemake_extra-3.13.3/bin/jobscript_master_debug.py {pid}'.format(pid=pid)
        subprocess.Popen(cmd, shell=True)


onsuccess:
    shutil.rmtree(".snakemake")
    print('Done.')


rule freesurfer:
    input:
        t1 = handle_storage('1000Brains_BIDS/{id}/{visit}/anat/{id}_{visit}_T1w.nii.gz'),
    output:
        pial = '1000Brains_derivatives/{id}/{visit}/anat/freesurfer/{id}_{visit}/surf/lh.pial',
    params:
        subjectdir = '1000Brains_derivatives/{id}/{visit}/anat/freesurfer',
#With the benchmark directive, Snakemake can be instructed to measure the wall clock time of a job
    benchmark:
        'Benchmarks/freesurfer-{id}_{visit}.txt'
    threads:
        1
#apparently the job wscheduler will read these resource specs
    resources:
        gpus = 0,
        mem = 2500,
        time = 1200
    shell:
        """
        module purge --force
        module use /usr/local/software/jureca/OtherStages
        module load Stages/Devel-2017a
        module load GCCcore/.5.4.0
        module load FreeSurfer/6.0.0
        export FS_FREESURFERENV_NO_OUTPUT=""
        source /usr/local/software/jureca/Stages/Devel-2017a/software/FreeSurfer/6.0.0-GCCcore-5.4.0/FreeSurferEnv.sh
        module use /data/inm1/mapping/software/2016b/modules
        module load FZJ/1.0.0
        module load Nano/2.7.4

        export SUBJECTS_DIR="{params.subjectdir}"

        recon-all -i {input.t1} -s {wildcards.id}_{wildcards.visit} -all
        """


rule brain_mask_orig:
    input:
        t1 = handle_storage('1000Brains_BIDS/{id}/{visit}/anat/{id}_{visit}_T1w.nii.gz'),
        tpm_gm = handle_storage('1000Brains_derivatives/{id}/{visit}/CAT/mri/p1{id}_{visit}-t1.nii'),
        tpm_wm = handle_storage('1000Brains_derivatives/{id}/{visit}/CAT/mri/p2{id}_{visit}-t1.nii'),
        tpm_csf = handle_storage('1000Brains_derivatives/{id}/{visit}/CAT/mri/p3{id}_{visit}-t1.nii'),
    output:
        brain_mask_pm = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_mask_pm.nii.gz',
        brain_mask = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_mask.nii.gz',
        brain = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain.nii.gz',
    params:
        threshold = 0.5
    benchmark:
        'Benchmarks/brain_mask_orig-{id}_{visit}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 5
    shell:
        SOURCE_COMMAND + """
        fslmaths {input.tpm_gm} -add {input.tpm_wm} -add {input.tpm_csf} {output.brain_mask_pm}
        fslmaths {output.brain_mask_pm} -thr {params.threshold} -bin -fillh {output.brain_mask} -odt char
        fslmaths {input.t1} -mas {output.brain_mask} {output.brain}
        """


rule brain_biascorrect:
    input:
        brain = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain.nii.gz'),
        brain_mask = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_mask.nii.gz'),
        tpm_wm = handle_storage('1000Brains_derivatives/{id}/{visit}/CAT/mri/p2{id}_{visit}-t1.nii'),
    output:
        brain_biascorrect = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_biascorrect.nii.gz',
        biasfield = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_biasfield.nii.gz',
    benchmark:
        'Benchmarks/brain_biascorrect-{id}_{visit}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 30
    shell:
        SOURCE_COMMAND + """
        N4BiasFieldCorrection -v 1 -d 3 -i {input.brain} -o [{output.brain_biascorrect},{output.biasfield}] --shrink-factor 2 \
                              --mask-image {input.brain_mask} --rescale-intensities 1 --weight-image {input.tpm_wm}
        """


rule align_t1w_colin:
    """
    Align the biascorrected T1w brain that is still in its original space (sagittal acquisition) to the colin template.
    The resulting transform can be combined with that from align_apm_t1w to transform ROIs from Colin space directly to dMRI space.
    """
    input:
        brain_mni = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_biascorrect.nii.gz'),
        colin = 'Projects/Anatomy_v23eintern/colin27T1_seg.nii.gz',
    output:
        affine = '1000Brains_derivatives/{id}/{visit}/anat/Colin/ANTs_T1w/{id}_{visit}_T1w_2_Colin_0GenericAffine.mat',
        warp = '1000Brains_derivatives/{id}/{visit}/anat/Colin/ANTs_T1w/{id}_{visit}_T1w_2_Colin_1Warp.nii.gz',
        invwarp = '1000Brains_derivatives/{id}/{visit}/anat/Colin/ANTs_T1w/{id}_{visit}_T1w_2_Colin_1InverseWarp.nii.gz',
        warped = '1000Brains_derivatives/{id}/{visit}/anat/Colin/ANTs_T1w/{id}_{visit}_T1w_2_Colin_Warped.nii.gz',
    params:
        ants_prefix = '1000Brains_derivatives/{id}/{visit}/anat/Colin/ANTs_T1w/{id}_{visit}_T1w_2_Colin_',
        converge_lin = '1000x500x250x100',
        converge_syn = '100x70x50x20',
        shrink_factors = '8x4x2x1'
    resources:
        gpus = 0,
        mem = 2500,
        time = 500,
    threads:
        cpu_count
    shell:
        SOURCE_COMMAND + """
        # antsRegistrationSyN.sh -d 3 \
        #                        -f {input.colin} \
        #                        -m {input.brain_mni} \
        #                        -o {params.ants_prefix} \
        #                        -n {threads} \
        #                        -t s \
        #                        -j 1
        export OMP_NUM_THREADS={threads}
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}
        antsRegistration --verbose 0 \
                         --dimensionality 3 \
                         --float 0 \
                         --output {params.ants_prefix} \
                         --interpolation Linear \
                         --use-histogram-matching 1 \
                         --winsorize-image-intensities [0.005,0.995] \
                         --initial-moving-transform [{input.colin},{input.brain_mni},1] \
                         --transform Rigid[0.1] \
                         --metric MI[{input.colin},{input.brain_mni},1,32,Regular,0.25] \
                         --convergence [{params.converge_lin},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox \
                         --transform Affine[0.1] \
                         --metric MI[{input.colin},{input.brain_mni},1,32,Regular,0.25] \
                         --convergence [{params.converge_lin},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox \
                         --transform SyN[0.1,3,0] \
                         --metric CC[{input.colin},{input.brain_mni},1,4] \
                         --convergence [{params.converge_syn},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox

        antsApplyTransforms --verbose 0 \
                            --dimensionality 3 \
                            --input-image-type 0 \
                            --input {input.brain_mni} \
                            --reference-image {input.colin} \
                            --output {output.warped} \
                            --interpolation Linear \
                            --transform {output.warp} \
                            --transform {output.affine} \
                            --default-value 0 \
                            --float 0
        """


rule align_t1w_MNI152:
    """
    Align the biascorrected T1w brain that is still in its original space (sagittal acquisition) to the MNI152 template (from FSL).
    The resulting transform can be combined with that from align_apm_t1w to transform ROIs from Colin space directly to dMRI space.
    """
    input:
        t1_brain = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_biascorrect.nii.gz'),
        mni_brain = 'Projects/1000BRAINS_Snakemake/MNI152_T1_1mm_brain.nii.gz',
    output:
        affine = '1000Brains_derivatives/{id}/{visit}/anat/MNI152/ANTs_T1w/{id}_{visit}_T1w_2_MNI152_0GenericAffine.mat',
        warp = '1000Brains_derivatives/{id}/{visit}/anat/MNI152/ANTs_T1w/{id}_{visit}_T1w_2_MNI152_1Warp.nii.gz',
        invwarp = '1000Brains_derivatives/{id}/{visit}/anat/MNI152/ANTs_T1w/{id}_{visit}_T1w_2_MNI152_1InverseWarp.nii.gz',
        warped = '1000Brains_derivatives/{id}/{visit}/anat/MNI152/ANTs_T1w/{id}_{visit}_T1w_2_MNI152_Warped.nii.gz',
    params:
        ants_prefix = '1000Brains_derivatives/{id}/{visit}/anat/MNI152/ANTs_T1w/{id}_{visit}_T1w_2_MNI152_',
        converge_lin = '1000x500x250x100',
        converge_syn = '100x70x50x20',
        shrink_factors = '8x4x2x1'
    resources:
        gpus = 0,
        mem = 2500,
        time = 500,
    threads:
        cpu_count
    shell:
        SOURCE_COMMAND + """
        # antsRegistrationSyN.sh -d 3 \
        #                        -f {input.mni_brain} \
        #                        -m {input.t1_brain} \
        #                        -o {params.ants_prefix} \
        #                        -n {threads} \
        #                        -t s \
        #                        -j 1
        export OMP_NUM_THREADS={threads}
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}
        antsRegistration --verbose 0 \
                         --dimensionality 3 \
                         --float 0 \
                         --output {params.ants_prefix} \
                         --interpolation Linear \
                         --use-histogram-matching 1 \
                         --winsorize-image-intensities [0.005,0.995] \
                         --initial-moving-transform [{input.mni_brain},{input.t1_brain},1] \
                         --transform Rigid[0.1] \
                         --metric MI[{input.mni_brain},{input.t1_brain},1,32,Regular,0.25] \
                         --convergence [{params.converge_lin},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox \
                         --transform Affine[0.1] \
                         --metric MI[{input.mni_brain},{input.t1_brain},1,32,Regular,0.25] \
                         --convergence [{params.converge_lin},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox \
                         --transform SyN[0.1,3,0] \
                         --metric CC[{input.mni_brain},{input.t1_brain},1,4] \
                         --convergence [{params.converge_syn},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox

        antsApplyTransforms --verbose 0 \
                            --dimensionality 3 \
                            --input-image-type 0 \
                            --input {input.t1_brain} \
                            --reference-image {input.mni_brain} \
                            --output {output.warped} \
                            --interpolation Linear \
                            --transform {output.warp} \
                            --transform {output.affine} \
                            --default-value 0 \
                            --float 0
        """


rule brain_mni_align:
    input:
        brain = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_biascorrect.nii.gz'),
        brain_mask_pm = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_mask_pm.nii.gz'),
        template = 'Projects/1000BRAINS_Snakemake/MNI152_T1_1mm_brain.nii.gz'
    output:
        trafo_dof12 = '1000Brains_derivatives/{id}/{visit}/anat/tmp/{id}_{visit}_T1w_brain_to_MNI_dof12.mat',
        trafo_dof06 = '1000Brains_derivatives/{id}/{visit}/anat/tmp/{id}_{visit}_T1w_brain_to_MNI_dof06.mat',
        brain_mni = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_1p00mm.nii.gz',
        brain_mask_pm_mni = '1000Brains_derivatives/{id}/{visit}/anat/tmp/{id}_{visit}_T1w_brain_mask_pm_1p00mm.nii.gz',
        brain_mask_mni = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_mask_1p00mm.nii.gz',
    params:
        threshold = 0.5
    benchmark:
        'Benchmarks/brain_mni_align-{id}_{visit}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 20
    shell:
        SOURCE_COMMAND + """
        flirt -in {input.brain} -ref {input.template} -usesqform -omat {output.trafo_dof12} -dof 12
        aff2rigid {output.trafo_dof12} {output.trafo_dof06}
        flirt -in {input.brain} -ref {input.template} -usesqform -init {output.trafo_dof06} -applyxfm -interp spline -out {output.brain_mni}
        flirt -in {input.brain_mask_pm} -ref {input.template} -usesqform -init {output.trafo_dof06} -applyxfm -interp spline -out {output.brain_mask_pm_mni}
        fslmaths {output.brain_mask_pm_mni} -thr {params.threshold} -bin -fillh {output.brain_mask_mni} -odt char
        fslmaths {output.brain_mni} -mas {output.brain_mask_mni} -max 0 {output.brain_mni}
        """


rule brain_resample_1p25mm:
    input:
        brain_mni = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_1p00mm.nii.gz'),
        brain_mask_pm_mni = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/tmp/{id}_{visit}_T1w_brain_mask_pm_1p00mm.nii.gz'),
    output:
        brain_mni = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_1p25mm.nii.gz',
        brain_mask_pm_mni = '1000Brains_derivatives/{id}/{visit}/anat/tmp/{id}_{visit}_T1w_brain_mask_pm_1p25mm.nii.gz',
        brain_mask_mni = '1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_mask_1p25mm.nii.gz',
    params:
        threshold = 0.5
    benchmark:
        'Benchmarks/brain_resample_1p25mm-{id}_{visit}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 5
    shell:
        SOURCE_COMMAND + """
        flirt -in {input.brain_mni} -out {output.brain_mni} -interp spline -ref {input.brain_mni} -applyisoxfm 1.25
        flirt -in {input.brain_mask_pm_mni} -out {output.brain_mask_pm_mni} -interp spline -ref {input.brain_mask_pm_mni} -applyisoxfm 1.25
        fslmaths {output.brain_mask_pm_mni} -thr {params.threshold} -bin -fillh {output.brain_mask_mni} -odt char
        fslmaths {output.brain_mni} -mas {output.brain_mask_mni} -max 0 {output.brain_mni}
        """


rule extract_b0:
    input:
        dwi = handle_storage('1000Brains_BIDS/{id}/{visit}/dwi/{id}_{visit}_{bval_numdir}_dwi.nii.gz'),
    output:
        b0 = '1000Brains_derivatives/{id}/{visit}/dwi/raw/{id}_{visit}_{bval_numdir}_b0.nii.gz',
    benchmark:
        'Benchmarks/extract_b0-{id}_{visit}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 5
    shell:
        SOURCE_COMMAND + """
        fslroi {input.dwi} {output.b0} 0 1
        """


rule align_b0_t1w:
    input:
        brain_mni = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_1p25mm.nii.gz'),
        b0_b1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/raw/{id}_{visit}_b1000-dMRI060_b0.nii.gz'),
        b0_b2700 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/raw/{id}_{visit}_b2700-dMRI120_b0.nii.gz'),
    output:
        b0_b1000 = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_b0_1p25mm.nii.gz',
        trafo_b1000 = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_b0_to_t1w.mat',
        b0_b2700 = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_b0_1p25mm.nii.gz',
        trafo_b2700 = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_b0_to_t1w.mat',
    benchmark:
        'Benchmarks/align_b0_t1w-{id}_{visit}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 5
    shell:
        SOURCE_COMMAND + """
        flirt -in {input.b0_b1000} -ref {input.brain_mni} -out {output.b0_b1000} -omat {output.trafo_b1000} -dof 6 -cost mutualinfo -interp spline
        flirt -in {input.b0_b2700} -ref {output.b0_b1000} -out {output.b0_b2700} -omat {output.trafo_b2700} -dof 6 -cost mutualinfo -interp spline
        """


rule align_b0_t1w_dMRI030:
    input:
        brain_mni = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_1p25mm.nii.gz'),
        b0_b1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/raw/{id}_{visit}_b1000-dMRI030_b0.nii.gz'),
    output:
        b0_b1000 = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI030_b0_1p25mm.nii.gz',
        trafo_b1000 = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI030_b0_to_t1w.mat',
    benchmark:
        'Benchmarks/align_b0_t1w_dMRI030-{id}_{visit}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 5
    shell:
        SOURCE_COMMAND + """
        flirt -in {input.b0_b1000} -ref {input.brain_mni} -out {output.b0_b1000} -omat {output.trafo_b1000} -dof 6 -cost mutualinfo -interp spline
        """


rule brain_mask_to_dwi:
    input:
        brain_mask_pm = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/tmp/{id}_{visit}_T1w_brain_mask_pm_1p25mm.nii.gz'),
        trafo = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{bval_numdir}_b0_to_t1w.mat'),
        b0 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/raw/{id}_{visit}_{bval_numdir}_b0.nii.gz'),
    output:
        trafo_inv = '1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_t1w_to_b0.mat',
        brain_mask_pm = '1000Brains_derivatives/{id}/{visit}/dwi/eddy/tmp/{id}_{visit}_{bval_numdir}_brain_mask_pm.nii.gz',
        brain_mask = '1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_brain_mask.nii.gz',
    params:
        threshold = 0.5
    benchmark:
        'Benchmarks/brain_mask_to_dwi-{id}_{visit}_{bval_numdir}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 2500,
        time = 5
    shell:
        SOURCE_COMMAND + """
        convert_xfm -omat {output.trafo_inv} -inverse {input.trafo}
        flirt -in {input.brain_mask_pm} -ref {input.b0} -out {output.brain_mask_pm} -init {output.trafo_inv} -applyxfm -interp spline
        fslmaths {output.brain_mask_pm} -thr {params.threshold} -bin -fillh {output.brain_mask} -odt char
        """


rule dwi_eddy:
    input:
        dmri = handle_storage('1000Brains_BIDS/{id}/{visit}/dwi/{id}_{visit}_{bval_numdir}_dwi.nii.gz'),
        bval = handle_storage('1000Brains_BIDS/{id}/{visit}/dwi/{id}_{visit}_{bval_numdir}_dwi.bval'),
        bvec = handle_storage('1000Brains_BIDS/{id}/{visit}/dwi/{id}_{visit}_{bval_numdir}_dwi.bvec'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_brain_mask.nii.gz'),
    output:
        dmri = '1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_dwi_eddy.nii.gz',
        index = '1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_eddy_index.txt',
        acqparams = '1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_eddy_acqparams.txt',
    params:
        eddy_basename = '1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_dwi_eddy',
        epi_readout_time = '0.0646',  # placeholder value
        eddy_exe = './eddy_5.0.11_openmp',
    benchmark:
        'Benchmarks/dwi_eddy-{id}_{visit}_{bval_numdir}.txt'
    threads:
        24
    resources:
        gpus = 0,
        mem = 65000,
        time = 120
    shell:
        SOURCE_COMMAND + """
        indx=""
        for ((i=1; i<=$(fslval {input.dmri} dim4); i+=1)); do indx="$indx 1"; done
        echo "$indx" > {output.index}
        echo "0 -1 0 {params.epi_readout_time}" > {output.acqparams}
        {params.eddy_exe} --imain={input.dmri} --mask={input.mask} --acqp={output.acqparams} --index={output.index} \
                          --bvecs={input.bvec} --bvals={input.bval} --out={params.eddy_basename} --repol --data_is_shelled
        """


rule dwi_remove_corrupted_volumes:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_dwi_eddy.nii.gz'),
        bval = handle_storage('1000Brains_BIDS/{id}/{visit}/dwi/{id}_{visit}_{bval_numdir}_dwi.bval'),
        bvec = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/eddy/{id}_{visit}_{bval_numdir}_dwi_eddy.eddy_rotated_bvecs'),
        qcinfo = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/qc/{id}_{visit}_{bval_numdir}_dwi_eddy.qcinfo'),
    output:
        dmri = '1000Brains_derivatives/{id}/{visit}/dwi/clean/{id}_{visit}_{bval_numdir}_dwi_clean.nii.gz',
        bval = '1000Brains_derivatives/{id}/{visit}/dwi/clean/{id}_{visit}_{bval_numdir}_dwi_clean.bval',
        bvec = '1000Brains_derivatives/{id}/{visit}/dwi/clean/{id}_{visit}_{bval_numdir}_dwi_clean.bvec',
        available = '1000Brains_derivatives/{id}/{visit}/dwi/clean/{id}_{visit}_{bval_numdir}_dwi_clean.available',
        remove_idx = '1000Brains_derivatives/{id}/{visit}/dwi/qc/{id}_{visit}_{bval_numdir}_dwi_remove_0based.txt',
    benchmark:
        'Benchmarks/dwi_remove_corrupted_volumes-{id}_{visit}_{bval_numdir}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 6500,
        time = 10
    shell:
        SOURCE_COMMAND + """
            # by default declare data set as missing (will be overwriritten on success)
            echo "0" > {output.available}

            # ensure that the set was rated as ok
            if [[ -z $(head -n 1 {input.qcinfo} | grep "SET_EVAL=1") ]]; then
                exit
            fi

            # create 0-based index of volumes to remove
            index=0
            for i in $(grep "VOL_EVAL=" {input.qcinfo} | cut -d '=' -f 2); do
                if [[ $i -gt 1 ]]; then
                    echo -ne "$index "
                fi
                let index=$index+1
            done > {output.remove_idx}
            echo "" >> {output.remove_idx}

            # remove corrupted volumes and corresponding entries in bvec and bval files
            fzj_dmri_cleandmri.sh -dmri {input.dmri} -bval {input.bval} -bvec {input.bvec} -remove {output.remove_idx} \
                                  -work /tmp/$(basename {input.qcinfo} .qcinfo) \
                                  -odmri {output.dmri} -obval {output.bval} -obvec {output.bvec}

            # verify that expected output files exist
            if [[ -s {output.dmri} ]]; then
                if [[ -s {output.bval} ]]; then
                    if [[ -s {output.bvec} ]]; then
                        echo "1" > {output.available}
                    fi
                fi
            fi
        """


rule resample_dwi_to_1p25mm:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/clean/{id}_{visit}_{bval_numdir}_dwi_clean.nii.gz'),
        bval = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/clean/{id}_{visit}_{bval_numdir}_dwi_clean.bval'),
        bvec = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/clean/{id}_{visit}_{bval_numdir}_dwi_clean.bvec'),
        brain_mni = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_1p25mm.nii.gz'),
        trafo = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{bval_numdir}_b0_to_t1w.mat'),
    output:
        dmri = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{bval_numdir}_dwi_1p25mm.nii.gz',
        bval = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{bval_numdir}_dwi_1p25mm.bval',
        bvec = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{bval_numdir}_dwi_1p25mm.bvec',
    benchmark:
        'Benchmarks/resample_dwi_to_1p25mm-{id}_{visit}_{bval_numdir}.txt'
    threads:
        1
    resources:
        gpus = 0,
        mem = 5500,
        time = 10
    shell:
        SOURCE_COMMAND + """
        flirt -in {input.dmri} -ref {input.brain_mni} -out {output.dmri} -init {input.trafo} -applyxfm -interp spline
        cp {input.bval} {output.bval}
        ${{FZJDIR}}/fzj_dmri_rotatebvecs.py {input.bvec} {input.trafo} {output.bvec}
        """


rule anisotropicpowermap:
    input:
        dmri120 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.nii.gz'),
        bval120 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.bval'),
        bvec120 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.bvec'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_mask_1p25mm.nii.gz'),
    output:
        apm = '1000Brains_derivatives/{id}/{visit}/dwi/APM/{id}_{visit}_b2700-dMRI120_dwi_APM.nii.gz',
    resources:
        gpus = 0,
        mem = 6500,
        time = 260,
    threads:
        2
    shell:
        """
        source /data/inm1/mapping/snakemake_3.13.3-2016b.source

        export OMP_NUM_THREADS={threads}
        python3 ./fzj_dmri_anisotropicpowermap.py \
                                  --attenuation \
                                  -j {threads} \
                                  {input.dmri120} \
                                  {input.bval120} \
                                  {input.bvec120} \
                                  {input.mask} \
                                  {output.apm}
        """


rule align_apm_t1w:
    input:
        apm = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/APM/{id}_{visit}_b2700-dMRI120_dwi_APM.nii.gz'),
        brain_mni = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_biascorrect.nii.gz'),
    output:
        affine = '1000Brains_derivatives/{id}/{visit}/dwi/APM/ANTs_T1w/{id}_{visit}_b2700-dMRI120_dwi_APM_2_T1w_0GenericAffine.mat',
        warp = '1000Brains_derivatives/{id}/{visit}/dwi/APM/ANTs_T1w/{id}_{visit}_b2700-dMRI120_dwi_APM_2_T1w_1Warp.nii.gz',
        invwarp = '1000Brains_derivatives/{id}/{visit}/dwi/APM/ANTs_T1w/{id}_{visit}_b2700-dMRI120_dwi_APM_2_T1w_1InverseWarp.nii.gz',
        warped = '1000Brains_derivatives/{id}/{visit}/dwi/APM/ANTs_T1w/{id}_{visit}_b2700-dMRI120_dwi_APM_2_T1w_Warped.nii.gz',
    params:
        ants_prefix = '1000Brains_derivatives/{id}/{visit}/dwi/APM/ANTs_T1w/{id}_{visit}_b2700-dMRI120_dwi_APM_2_T1w_',
        converge_lin = '1000x500x250x100',
        converge_syn = '100x70x50x20',
        shrink_factors = '8x4x2x1'
    resources:
        gpus = 0,
        mem = 2500,
        time = 500,
    threads:
        cpu_count
    shell:
        SOURCE_COMMAND + """
        # antsRegistrationSyN.sh -d 3 \
        #                        -f {input.brain_mni} \
        #                        -m {input.apm} \
        #                        -o {params.ants_prefix} \
        #                        -n {threads} \
        #                        -t s \
        #                        -j 1
        export OMP_NUM_THREADS={threads}
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}
        antsRegistration --verbose 0 \
                         --dimensionality 3 \
                         --float 0 \
                         --output {params.ants_prefix} \
                         --interpolation Linear \
                         --use-histogram-matching 1 \
                         --winsorize-image-intensities [0.005,0.995] \
                         --initial-moving-transform [{input.brain_mni},{input.apm},1] \
                         --transform Rigid[0.1] \
                         --metric MI[{input.brain_mni},{input.apm},1,32,Regular,0.25] \
                         --convergence [{params.converge_lin},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox \
                         --transform Affine[0.1] \
                         --metric MI[{input.brain_mni},{input.apm},1,32,Regular,0.25] \
                         --convergence [{params.converge_lin},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox \
                         --transform SyN[0.1,3,0] \
                         --metric CC[{input.brain_mni},{input.apm},1,4] \
                         --convergence [{params.converge_syn},1e-6,10] \
                         --shrink-factors {params.shrink_factors} \
                         --smoothing-sigmas 3x2x1x0vox

        antsApplyTransforms --verbose 0 \
                            --dimensionality 3 \
                            --input-image-type 0 \
                            --input {input.apm} \
                            --reference-image {input.brain_mni} \
                            --output {output.warped} \
                            --interpolation Linear \
                            --transform {output.warp} \
                            --transform {output.affine} \
                            --default-value 0 \
                            --float 0
        """


rule transform_tpms:
    input:
        tpm_gm = handle_storage('1000Brains_derivatives/{id}/{visit}/CAT/mri/p1{id}_{visit}-t1.nii'),
        tpm_wm = handle_storage('1000Brains_derivatives/{id}/{visit}/CAT/mri/p2{id}_{visit}-t1.nii'),
        tpm_csf = handle_storage('1000Brains_derivatives/{id}/{visit}/CAT/mri/p3{id}_{visit}-t1.nii'),
        apm = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/APM/{id}_{visit}_b2700-dMRI120_dwi_APM.nii.gz'),
        affine = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/APM/ANTs_T1w/{id}_{visit}_b2700-dMRI120_dwi_APM_2_T1w_0GenericAffine.mat'),
        invwarp = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/APM/ANTs_T1w/{id}_{visit}_b2700-dMRI120_dwi_APM_2_T1w_1InverseWarp.nii.gz'),
    output:
        tpm_gm = '1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_gm_1p25mm.nii.gz',
        tpm_wm = '1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_wm_1p25mm.nii.gz',
        tpm_csf = '1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_csf_1p25mm.nii.gz',
        brain_mask_nl = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz',
    params:
        threshold = 0.5
    resources:
        gpus = 0,
        mem = 2500,
        time = 200,
    threads:
        1
    shell:
        SOURCE_COMMAND + """
        antsApplyTransforms --verbose 0 \
                            --dimensionality 3 \
                            --input-image-type 0 \
                            --input {input.tpm_gm} \
                            --reference-image {input.apm} \
                            --output {output.tpm_gm} \
                            --interpolation Linear \
                            --transform [{input.affine},1] \
                            --transform {input.invwarp} \
                            --default-value 0 \
                            --float 0

        antsApplyTransforms --verbose 0 \
                            --dimensionality 3 \
                            --input-image-type 0 \
                            --input {input.tpm_wm} \
                            --reference-image {input.apm} \
                            --output {output.tpm_wm} \
                            --interpolation Linear \
                            --transform [{input.affine},1] \
                            --transform {input.invwarp} \
                            --default-value 0 \
                            --float 0

        antsApplyTransforms --verbose 0 \
                            --dimensionality 3 \
                            --input-image-type 0 \
                            --input {input.tpm_csf} \
                            --reference-image {input.apm} \
                            --output {output.tpm_csf} \
                            --interpolation Linear \
                            --transform [{input.affine},1] \
                            --transform {input.invwarp} \
                            --default-value 0 \
                            --float 0

        fslmaths {output.tpm_gm} -add {output.tpm_wm} -add {output.tpm_csf} -thr {params.threshold} -bin -fillh {output.brain_mask_nl} -odt char
        """


rule noddi_prtcl:
    input:
        bval_1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bval'),
        bvec_1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bvec'),
        bval_2700 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.bval'),
        bvec_2700 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.bvec'),
    output:
        prtcl = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_noddi.prtcl',
    resources:
        gpus = 0,
        mem = 1000,
        time = 5
    threads:
        1
    run:
        import numpy as np

        bvals_1000 = np.genfromtxt(input.bval_1000, delimiter=' ')
        bvecs_1000 = np.genfromtxt(input.bvec_1000, delimiter=' ').T

        bvals_2700 = np.genfromtxt(input.bval_2700, delimiter=' ')
        bvecs_2700 = np.genfromtxt(input.bvec_2700, delimiter=' ').T

        prtcl_1000 = np.vstack([bvecs_1000[:, 0],                # gx
                                bvecs_1000[:, 1],                # gy
                                bvecs_1000[:, 2],                # gz
                                bvals_1000 * 10**(6),            # b
                                [6.3] * bvals_1000.shape[0],     # TR
                                [81e-3] * bvals_1000.shape[0],   # TE
                                [40e-3] * bvals_1000.shape[0]])  # maxG

        prtcl_2700 = np.vstack([bvecs_2700[:, 0],                # gx
                                bvecs_2700[:, 1],                # gy
                                bvecs_2700[:, 2],                # gz
                                bvals_2700 * 10**(6),            # b
                                [8.0] * bvals_2700.shape[0],     # TR
                                [112e-3] * bvals_2700.shape[0],  # TE
                                [40e-3] * bvals_2700.shape[0]])  # maxG

        prtcl = np.hstack([prtcl_1000, prtcl_2700])

        np.savetxt(output.prtcl, prtcl.T, fmt='%.10e',
                   delimiter='\t', newline='\n',
                   header='gx,gy,gz,b,TR,TE,maxG',
                   footer='', comments='#')


rule mdt_init:
    output:
        model = str(Path.home()) + '/.mdt/{mdt_version}/components/standard/cascade_models/NODDI.py'
    resources:
        gpus = 0,
        mem = 600,
        time = 10
    threads:
        1
    shell:
        SOURCE_COMMAND + """
        module load MDT/{wildcards.mdt_version}

        mdt-init-user-settings
        """


rule noddi_merge_and_noise_std:
    input:
        dmri_1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.nii.gz'),
        dmri_2700 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.nii.gz'),
        prtcl = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_noddi.prtcl'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz'),
        model = str(Path.home()) + '/.mdt/0.9.31/components/standard/cascade_models/NODDI.py',
    output:
        dmri = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700.nii.gz',
        noisestd = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700_noise_std.txt',
    resources:
        gpus = 4,
        mem = 5500,
        time = 10,
    threads:
        1
    shell:
        SOURCE_COMMAND + """
        fslmerge -t {output.dmri} {input.dmri_1000} {input.dmri_2700}

        module load MDT/0.9.31

        mdt-estimate-noise-std {output.dmri} {input.prtcl} {input.mask} > {output.noisestd}

        if [[ ! -s {output.noisestd} ]]; then
            if [[ -e {output.noisestd} ]]; then
                rm -f {output.noisestd}
            fi
            exit 1
        fi
        """


rule noddi_mdt:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700.nii.gz'),
        prtcl = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_noddi.prtcl'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz'),
        model = str(Path.home()) + '/.mdt/0.9.31/components/standard/cascade_models/NODDI.py',
        noisestd = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700_noise_std.txt'),
    output:
        icvf = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700_icvf.nii.gz',
        ecvf = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700_ecvf.nii.gz',
        csfvf = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700_csfvf.nii.gz',
        odi = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI/{id}_{visit}_dwi_b1000-b2700_odi.nii.gz',
    params:
        outdir = '1000Brains_derivatives/{id}/{visit}/dwi/NODDI',
        # noisestd = 50,
    resources:
        gpus = 4,
        mem = 1000,
        time = 120,
    log:
        'Logs/noddi_mdt-{id}_{visit}_b1000-b2700.txt'
    threads:
        24
    shell:
        SOURCE_COMMAND + """
        module load MDT/0.9.31
        mdt-model-fit -o {params.outdir} \
                      -n $(echo -ne $(cat {input.noisestd})) \
                      --double \
                      --use-cascade-subdir \
                      "NODDI (Cascade|fixed)" \
                      {input.dmri} \
                      {input.prtcl} \
                      {input.mask} > {log} 2> {log}
        mkdir -p {params.outdir}
        cp "{params.outdir}/NODDI (Cascade|fixed)/NODDI/w_ic.w.nii.gz" {output.icvf}
        cp "{params.outdir}/NODDI (Cascade|fixed)/NODDI/w_ec.w.nii.gz" {output.ecvf}
        cp "{params.outdir}/NODDI (Cascade|fixed)/NODDI/w_csf.w.nii.gz" {output.csfvf}
        cp "{params.outdir}/NODDI (Cascade|fixed)/NODDI/ODI.nii.gz" {output.odi}
        """


rule diffusion_tensor:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{dwi}_dwi_1p25mm.nii.gz'),
        bval = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{dwi}_dwi_1p25mm.bval'),
        bvec = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_{dwi}_dwi_1p25mm.bvec'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz'),
    output:
        fa = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_FA.nii.gz',
        md = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_MD.nii.gz',
        ad = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_AD.nii.gz',
        rd = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_RD.nii.gz',
        l1 = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_L1.nii.gz',
        l2 = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_L2.nii.gz',
        l3 = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_L3.nii.gz',
        v1 = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_V1.nii.gz',
        v2 = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_V2.nii.gz',
        v3 = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm_V3.nii.gz',
    params:
        basename = '1000Brains_derivatives/{id}/{visit}/dwi/tensor/{id}_{visit}_{dwi}_dwi_Tensor_1p25mm',
    resources:
        gpus = 0,
        mem = 1000,
        time = 5,
    threads:
        1
    shell:
        SOURCE_COMMAND + """
        dtifit -k {input.dmri} -m {input.mask} -b {input.bval} -r {input.bvec} -o {params.basename} --wls

        cp {output.l1} {output.ad}
        fslmaths {output.l2} -add {output.l3} -div 2 {output.rd}
        """


rule merge_and_normalize_dwi:
    input:
        dmri_1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.nii.gz'),
        bval_1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bval'),
        bvec_1000 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bvec'),
        dmri_2700 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.nii.gz'),
        bval_2700 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.bval'),
        bvec_2700 = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_b2700-dMRI120_dwi_1p25mm.bvec'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz'),
    output:
        dmri = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.nii.gz',
        bval = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bval',
        bvec = '1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bvec',
    params:
        work = temp('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/work'),
        mask = temp('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/work/merged_brain_mask.nii.gz'),
    resources:
        gpus = 0,
        mem = 6000,
        time = 30
    threads:
        4
    shell:
        SOURCE_COMMAND + """
        fzj_dmri_mergeandnormalizeb0.sh -dmri1 {input.dmri_1000}  -bval1 {input.bval_1000}  -bvec1 {input.bvec_1000} -mask1 {input.mask} \
                                        -dmri2 {input.dmri_2700} -bval2 {input.bval_2700} -bvec2 {input.bvec_2700} -mask2 {input.mask} \
                                        -odmri {output.dmri} -obval {output.bval} -obvec {output.bvec} -omask {params.mask}\
                                        -work {params.work} -parallel {threads}
        """


rule compute_csd_kernel:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.nii.gz'),
        bval = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bval'),
        bvec = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bvec'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz'),
    output:
        work = temp('1000Brains_derivatives/{id}/{visit}/dwi/csd/work_kernel'),
        kernel = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_csd-kernel.txt'
    resources:
        gpus = 0,
        mem = 6000,
        time = 90
    threads:
        24
    shell:
        SOURCE_COMMAND + """
        fzj_dmri_csd.sh -dmri {input.dmri} -bval {input.bval} -bvec {input.bvec} \
                        -mask {input.mask} -lmax 0,8,8 -work {output.work} \
                        -oresponse {output.kernel} -response-only 1 -signalatt 1 \
                        -parallel {threads} -parproc {threads}
        """


rule csd_kernel:
    output:
        kernel = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_average_csd_kernel.nii.gz',
    resources:
        gpus = 0,
        mem = 100,
        time = 15,
    threads:
        1
    shell:
        SOURCE_COMMAND + """
        echo "1417.964 0 0 0 0" > {output.kernel}
        echo "879.168 -290.8096 45.5356 -4.80028 0.3816128" >> {output.kernel}
        echo "488.092 -325.0944 121.6348 -32.39368 6.6732" >> {output.kernel}
        """


rule compute_csd_local_model:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.nii.gz'),
        bval = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bval'),
        bvec = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bvec'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz'),
        kernel = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_average_csd_kernel.nii.gz'),
    output:
        csd = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_CSD.nii.gz',
        afd = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_CSD_afd.nii.gz',
        disp = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_CSD_disp.nii.gz',
        peak = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_CSD_peak.nii.gz',
        dir = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_CSD_dir.nii.gz',
        fa = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_CSD_fa.nii.gz',
    params:
        work = temp('1000Brains_derivatives/{id}/{visit}/dwi/csd/work_csd'),
    log:
        log = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_CSD.log',
    resources:
        gpus = 0,
        mem = 6000,
        time = 60
    threads:
        24
    shell:
        SOURCE_COMMAND + """
        fzj_dmri_csd.sh -dmri {input.dmri} -bval {input.bval} -bvec {input.bvec} \
                        -mask {input.mask} -lmax 0,8,8 -response {input.kernel} \
                        -csd {output.csd} -afd {output.afd} -disp {output.disp} \
                        -peak {output.peak} -dir {output.dir} -log {log.log} \
                        -fa {output.fa} \
                        -work {params.work} -parallel {threads} -parproc {threads}
        """


rule segmentation_5tt:
    input:
        brain = handle_storage('1000Brains_derivatives/{id}/{visit}/anat/{id}_{visit}_T1w_brain_1p25mm.nii.gz'),
        tpm_gm = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_gm_1p25mm.nii.gz'),
        tpm_wm = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_wm_1p25mm.nii.gz'),
        tpm_csf = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_csf_1p25mm.nii.gz'),
    output:
        tpm5tt = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_5tt_1p25mm.nii.gz',
        L_Accu = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_L_Accu_1p25mm.mif',
        R_Accu = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_R_Accu_1p25mm.mif',
        L_Caud = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_L_Caud_1p25mm.mif',
        R_Caud = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_R_Caud_1p25mm.mif',
        L_Pall = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_L_Pall_1p25mm.mif',
        R_Pall = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_R_Pall_1p25mm.mif',
        L_Puta = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_L_Puta_1p25mm.mif',
        R_Puta = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_R_Puta_1p25mm.mif',
        L_Thal = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_L_Thal_1p25mm.mif',
        R_Thal = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_R_Thal_1p25mm.mif',
    params:
        work = temp('1000Brains_derivatives/{id}/{visit}/dwi/csd/work_5tt'),
        scs = "L_Accu R_Accu L_Caud R_Caud L_Pall R_Pall L_Puta R_Puta L_Thal R_Thal"
    log:
        log = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_5tt.log',
    resources:
        gpus = 0,
        mem = 6000,
        time = 60
    threads:
        1
    shell:
        SOURCE_COMMAND + """
        old_wd=$(pwd)
        if [[ -d {params.work} ]]; then
            rm -fr {params.work}
        fi
        mkdir -p {params.work}
        cd {params.work}
        cp {input.brain} T1.nii.gz
        run_first_all -m none -s $(echo {params.scs} | sed -e "s/ /,/g") -i T1.nii.gz -o first -b
        cd ${{old_wd}}
        sc_mif=""
        for sc in {params.scs}; do
            EXE="meshconvert -force {params.work}/first-${{sc}}_first.vtk {params.work}/${{sc}}.vtk -transform_first2real {input.brain}"
            echo "$EXE"
            eval "$EXE"
            filename=$(echo "{output.L_Accu}" | sed -e "s/_L_Accu_1p25mm.mif/_${{sc}}_1p25mm.mif/")
            EXE="mesh2pve -force {params.work}/${{sc}}.vtk {input.brain} ${{filename}}"
            echo "$EXE"
            eval "$EXE"
            sc_mif="${{sc_mif}} ${{filename}}"
        done
        mrmath ${{sc_mif}} sum - | mrcalc - 1.0 -min {params.work}/all_sgms.mif
        mrthreshold {input.tpm_wm} - -abs 0.001 | maskfilter - connect - -connectivity | mrcalc 1 - 1 -gt -sub {params.work}/remove_unconnected_wm_mask.mif -datatype bit
        mrcalc {input.tpm_csf} {params.work}/remove_unconnected_wm_mask.mif -mult {params.work}/csf.mif
        mrcalc 1.0 {params.work}/csf.mif -sub {params.work}/all_sgms.mif -min {params.work}/sgm.mif
        mrcalc 1.0 {params.work}/csf.mif {params.work}/sgm.mif -add -sub {input.tpm_gm} {input.tpm_wm} -add -div {params.work}/multiplier.mif
        mrcalc {params.work}/multiplier.mif -finite {params.work}/multiplier.mif 0.0 -if {params.work}/multiplier_noNAN.mif
        mrcalc {input.tpm_gm} {params.work}/multiplier_noNAN.mif -mult {params.work}/remove_unconnected_wm_mask.mif -mult {params.work}/cgm.mif
        mrcalc {input.tpm_wm} {params.work}/multiplier_noNAN.mif -mult {params.work}/remove_unconnected_wm_mask.mif -mult {params.work}/wm.mif
        mrcalc 0 {params.work}/wm.mif -min {params.work}/path.mif
        mrcat {params.work}/cgm.mif {params.work}/sgm.mif {params.work}/wm.mif {params.work}/csf.mif {params.work}/path.mif - -axis 3 | mrconvert - {params.work}/combined_precrop.mif -stride +2,+3,+4,+1
        mrmath {params.work}/combined_precrop.mif sum - -axis 3 | mrthreshold - - -abs 0.5 | mrcrop -force {params.work}/combined_precrop.mif {output.tpm5tt} -mask -
        """


# rule compose_5tt:
#     input:
#         tpm_gm = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_gm_1p25mm.nii.gz'),
#         tpm_wm = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_wm_1p25mm.nii.gz'),
#         tpm_csf = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/TPM/{id}_{visit}_tpm_csf_1p25mm.nii.gz'),
#     output:
#         tpm5tt = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_5tt.nii.gz',
#     benchmark:
#         'Benchmarks/compose_5tt-{id}_{visit}.txt'
#     threads:
#         1
#     resources:
#         gpus = 0,
#         mem = 3000,
#         time = 5
#     run:
#         import numpy as np
#         import nibabel as nib
#
#         p1 = nib.load(input.tpm_gm).get_data().astype(np.float)
#         p2 = nib.load(input.tpm_wm).get_data().astype(np.float)
#         p3 = nib.load(input.tpm_csf).get_data().astype(np.float)
#         zeros = np.zeros_like(p1).astype(np.float)
#
#         data = np.stack([p1, zeros, p2, p3, zeros], axis=3)
#
#         img = nib.Nifti1Image(data, nib.load(input.tpm_gm).affine)
#         nib.save(img, output.tpm5tt)


rule compute_5tt_kernel:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.nii.gz'),
        bval = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bval'),
        bvec = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bvec'),
        tpm5tt = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_5tt_1p25mm.nii.gz'),
    output:
        kernel_gm = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_kernel_gm.txt',
        kernel_wm = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_kernel_wm.txt',
        kernel_csf = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_kernel_csf.txt',
    benchmark:
        'Benchmarks/compute_5tt_kernel-{id}_{visit}.txt'
    threads:
        24
    resources:
        gpus = 0,
        mem = 6000,
        time = 600
    shell:
        SOURCE_COMMAND + """
        dwi2response msmt_5tt -fslgrad {input.bvec} {input.bval} \
                                       {input.dmri} \
                                       {input.tpm5tt} \
                                       {output.kernel_wm} \
                                       {output.kernel_gm} \
                                       {output.kernel_csf}
        """


rule compute_5tt_csd:
    input:
        dmri = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.nii.gz'),
        bval = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bval'),
        bvec = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/NormB0/{id}_{visit}_b1000-dMRI060_dwi_1p25mm.bvec'),
        mask = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/1p25mm/{id}_{visit}_brain_mask_1p25mm.nii.gz'),
        ak_gm = handle_storage('1000Brains_derivatives/Average_Kernel_1000BRAINS/kernel_average_gm.txt'),
        ak_wm = handle_storage('1000Brains_derivatives/Average_Kernel_1000BRAINS/kernel_average_wm.txt'),
        ak_csf = handle_storage('1000Brains_derivatives/Average_Kernel_1000BRAINS/kernel_average_csf.txt'),
    output:
        csd_gm = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_csd_gm_1p25mm.nii.gz',
        csd_wm = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_csd_wm_1p25mm.nii.gz',
        csd_csf = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_csd_csf_1p25mm.nii.gz',
        vf = '1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_csd_vf_1p25mm.mif',
    benchmark:
        'Benchmarks/compute_5tt_csd-{id}_{visit}.txt'
    threads:
        24
    resources:
        gpus = 0,
        mem = 6000,
        time = 10
    shell:
        SOURCE_COMMAND + """
        # compute csd model
        dwi2fod msmt_csd -force \
                         -shell 0,1000,2700 \
                         -lmax 0,8,8 \
                         -nthreads {threads} \
                         -mask {input.mask} \
                         -fslgrad {input.bvec} {input.bval} \
                         {input.dmri} \
                         {input.ak_wm} {output.csd_wm} \
                         {input.ak_gm} {output.csd_gm} \
                         {input.ak_csf} {output.csd_csf}

        # merge results
        mrconvert -coord 3 0 {output.csd_wm} - | mrcat {output.csd_csf} {output.csd_gm} - {output.vf}
        """


rule tractography:
    input:
        tpm5tt = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_5tt_1p25mm.nii.gz'),
        csd_wm = handle_storage('1000Brains_derivatives/{id}/{visit}/dwi/csd/{id}_{visit}_csd_wm_1p25mm.nii.gz'),
    output:
        streamlines = '1000Brains_derivatives/{id}/{visit}/dwi/tracts/{id}_{visit}_tracks_act_{num}-{part}.tck',
        seeds = '{id}/{visit}/dMRI/Tractography/{id}_{visit}-seeds_act_{num}-{part}.txt',
    benchmark:
        'Benchmarks/tractography-{id}_{visit}-{num}-{part}.txt'
    threads:
        24
    resources:
        gpus = 0,
        mem = 6000,
        time = 800
    shell:
        SOURCE_COMMAND + """
        tckgen -quiet {input.csd_wm} {output.streamlines} \
               -act {input.tpm5tt} \
               -backtrack \
               -crop_at_gmwmi \
               -seed_dynamic {input.csd_wm} \
               -maxlength 250 \
               -number {wildcards.num} \
               -cutoff 0.06 \
               -output_seeds {output.seeds} \
               -nthreads {threads}
        """
