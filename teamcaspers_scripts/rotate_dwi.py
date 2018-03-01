# coding: utf-8

from __future__ import print_function, division

import argparse
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.core.geometry import cart2sphere
from dipy.core.sphere import HemiSphere
import dipy.reconst.dti as dti

def main():
    
    description='This script...'
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('-data',       help=' [IN] ', required=True)    
    parser.add_argument('-bval',       help=' [IN] ', required=True)    
    parser.add_argument('-bvec',       help=' [IN] ', required=True)    
    parser.add_argument('-mask',       help=' [IN] ', required=True)    
    parser.add_argument('-ref_bval',   help=' [IN] ', required=True)    
    parser.add_argument('-ref_bvec',   help=' [IN] ', required=True)    
    parser.add_argument('-order',      help=' [IN] ', default=6, type=int)    
    parser.add_argument('-out',        help='[OUT] ', required=True)
    parser.add_argument('--verbose', '-v', help='be verbose', action='store_true')

    args = parser.parse_args()
    
    dmri_file_name = args.data
    mask_file_name = args.mask
    bval_file_name = args.bval
    bvec_file_name = args.bvec
    
    ref_bval_file_name = args.ref_bval
    ref_bvec_file_name = args.ref_bvec
    
    out_file_name = args.out
    sh_order = args.order
    verbose = args.verbose
    
    
    # load the file's b-values and b-vectors
    if verbose:
        print('Loading gradient table')
    gtab = gradient_table(bval_file_name, bvec_file_name, atol=0.1)
    b0s_mask = gtab.b0s_mask
    #dwi_mask = np.invert(gtab.b0s_mask)
    
    # round b-values
    gtab.bvals = np.round(gtab.bvals / 100.0) * 100 
    # extract unique b-values > 0
    unique_bvalues = np.unique(gtab.bvals).tolist()[1:]
    
    #bvecs_dwi = gtab.bvecs[dwi_mask]
    
    bvec_spheres = []
    for b in unique_bvalues:
        r, theta, phi = cart2sphere(gtab.bvecs[gtab.bvals == b][:, 0], 
                                    gtab.bvecs[gtab.bvals == b][:, 1], 
                                    gtab.bvecs[gtab.bvals == b][:, 2])
        bvec_spheres.append(HemiSphere(theta=theta, phi=phi))
    
    # load the data
    if verbose:
        print('Loading dMRI data')
    affine = nib.load(dmri_file_name).get_affine()
    data = nib.load(dmri_file_name).get_data()
        
    # load the mask
    if verbose:
        print('Loading mask')
    mask = nib.load(mask_file_name).get_data() > 0
    
    # compute the b0's mean
    if verbose:
        print('Computing b0 average')
    average_b0 = np.mean(data[..., b0s_mask], axis=3)
    
    # compute the spherical harmonics
    if verbose:
        print('Computing spherical harmonics')
    shm = [sf_to_sh(data[..., gtab.bvals == b], bvec_spheres[i], sh_order=sh_order, basis_type=None, smooth=0.0) for i, b in enumerate(unique_bvalues)] 
    
    # compute diffusion tensor
    if verbose:
        print('Computing eigenvectors of diffusion tensor')
    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask=mask)
    evecs = tenfit.evecs

    # load the reference b-values and b-vectors
    if verbose:
        print('Loading reference gradient table')
    ref_gtab = gradient_table(ref_bval_file_name, ref_bvec_file_name)
    ref_b0s_mask = ref_gtab.b0s_mask
    ref_dwi_mask = np.invert(ref_gtab.b0s_mask)
    
    # round b-values
    ref_gtab.bvals = np.round(ref_gtab.bvals / 100.0) * 100 
    
    #ref_bvec_spheres = []
    #for b in unique_bvalues:
    #    r, theta, phi = cart2sphere(ref_gtab.bvecs[ref_gtab.bvals == b][:, 0], 
    #                                ref_gtab.bvecs[ref_gtab.bvals == b][:, 1], 
    #                                ref_gtab.bvecs[ref_gtab.bvals == b][:, 2])
    #    ref_bvec_spheres.append(HemiSphere(theta=theta, phi=phi))    
    
    
    # rotate the data
    if verbose:
        print('Rotating the data in every voxel of the mask')
    data_interp = []
    for i, b in enumerate(unique_bvalues):
        data_i = np.zeros((data.shape[0], data.shape[1], data.shape[2], 
                           np.sum(ref_gtab.bvals == b)), dtype=np.float)
        u = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float)
        shm_i = shm[i]
        ref_bvecs_dwi = ref_gtab.bvecs[ref_gtab.bvals == b]
        
        # iterate over all voxels
        for cc in np.ndindex(data.shape[0], data.shape[1], data.shape[2]):
            # only process voxels within mask
            if mask[cc]:
                v = evecs[cc]
                m = u.dot(np.linalg.pinv(v))
                d = np.dot(ref_bvecs_dwi, m)
                #d = ref_bvecs_dwi
                r, theta, phi = cart2sphere(d[:, 0], d[:, 1], d[:, 2])
                sphere = HemiSphere(theta=theta, phi=phi)
                sf = sh_to_sf(shm_i[cc], sphere=sphere, sh_order=sh_order, basis_type=None)
                data_i[cc[0], cc[1], cc[2], :] = sf
        data_interp.append(data_i)            
    
    #data_interp = sh_to_sf(shm, sphere=ref_bvec_sphere, 
    #                       sh_order=sh_order, basis_type=None)
    
    # save the interpolated data
    #  create new data array
    if verbose:
        print('Composing rotated diffusion weighted and averages b0 data')
    data_new = np.zeros((data.shape[0], data.shape[1], data.shape[2], ref_gtab.bvals.shape[0]), dtype=np.float32)
    #  fill the b0 volumes with data from average b0
    data_new[..., ref_b0s_mask] = np.stack([average_b0 for i in range(np.sum(ref_b0s_mask))], axis=3)
    #  fill the dwi data volumes with the rotated dwi data
    for i, b in enumerate(unique_bvalues):
        data_new[..., ref_gtab.bvals == b] = data_interp[i]
    
    #  save the data
    nifti_img = nib.Nifti1Image(data_new, affine)
    nib.save(nifti_img, out_file_name)

if __name__ == '__main__':
    main()
    