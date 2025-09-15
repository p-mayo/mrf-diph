# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:46:27 2023

@author: pm15334
"""

import os
import glob
import mat73
import torch
import numpy as np
import scipy.io as sio

import utils

from mrf_processing.utils_acquisition2d import NUFFT_Pytorch, load_traj_basis_dcf_2dscans

deli_slices = {
    256 : [[140, 170], # 0
           [137, 167], # 1
           [140, 170], # 2
           [145, 175], # 3
           [140, 170], # 4
           [152, 182], # 5
           [150, 180], # 6
           [155, 185], # 7
           [140, 170], # 8
           [150, 180], # 9
           [140, 170], # 10
           [150, 180], # 11
           ],
}

def load_data(data_path, to_tensor=True, simplify_cells=True):
    try:
        file_data = sio.loadmat(data_path, simplify_cells=simplify_cells)
    except NotImplementedError:
        print('It\'s a Matlab 7.3, attempting to open with mat73 library')
        file_data = mat73.loadmat(data_path)  # Required for Matlab v7.3 files
    keys = list(file_data.keys())
    for k in keys:
        if k.startswith('__'):
            del file_data[k]
        else:
            if to_tensor:
                file_data = numpy2tensor(file_data)
    return file_data


def numpy2tensor(data):
    if type(data) == dict:
        for k in data.keys():
            data[k] = numpy2tensor(data[k])
    elif type(data) == np.ndarray:
        data = torch.from_numpy(data)
    return data


def get_tsmis_reference_from_qmaps(dict_path, qmaps_dir, tsmi_output_dir, qmaps_output_dir, k_svd=5, fisp_cut=0,
                                   mask_threshold = 0.):
    os.makedirs(tsmi_output_dir, exist_ok=True)
    os.makedirs(qmaps_output_dir, exist_ok=True)

    for mtx in deli_slices.keys():
        for ri, slice_range in enumerate(deli_slices[mtx]):
            print(f'Processing mtx{mtx} vol {ri}\n\tSlices ', end='')
            f = f'pyrecon_mtx{mtx}_mrfadmm_bis_dm_lam1.00e-04_it5_cgit10_vol{ri}'
            qmaps = load_data(os.path.join(qmaps_dir, f'{f}.mat'))['out']['q_rec']
            mask = torch.abs(qmaps[2] + 1j*qmaps[3]) > mask_threshold

            qmaps = qmaps[:, :, 13:-13, 13:-13]
            mask = mask[ :, 13:-13, 13:-13]

            for s, slice_num in enumerate(range(slice_range[0], slice_range[1])):
                print(f'{slice_num + 1}, ', end='')
                tsmi_ref = utils.qmaps_to_tsmis_kdtree_svd(qmaps[:, slice_num, :, :].unsqueeze(0), dict_path=dict_path)
                if tsmi_ref.shape[1] > k_svd:
                    tsmi_ref = tsmi_ref[:, :k_svd, :, :]

                fname = f'qmaps_vol{ri}_slice{s}'
                output_file = os.path.join(qmaps_output_dir, f'{fname}.npz')
                np.savez(output_file, qmaps=qmaps[:, slice_num, :, :].squeeze())

                fname = fname.replace('qmaps', 'tsmi_ref')
                output_file = os.path.join(tsmi_output_dir, f'{fname}_cut{fisp_cut}.npz')
                np.savez(output_file, tsmi_ref=tsmi_ref)

                fname = fname.replace('tsmi_ref', 'mask')
                output_file = os.path.join(qmaps_output_dir, f'{fname}_th{mask_threshold}.npz')
                np.savez(output_file, mask=mask[slice_num, :, :].squeeze())
            print()

def get_kspace_from_reference_tsmi(tsmi_dir, kspace_output_dir, cut, device, nufft_operator):
    tsmi_paths = glob.glob(os.path.join(tsmi_dir, f'*_cut{cut}.npz'))
    os.makedirs(kspace_output_dir, exist_ok=True)

    for tsmi_path in tsmi_paths:
        tsmi = torch.from_numpy(np.load(tsmi_path)['tsmi_ref'])
        tsmi = tsmi.to(device)

        kspace = nufft_operator.fwd(tsmi.to(device), sens=None)
        print(kspace.shape, torch.view_as_real(kspace).max())
        kspace = kspace + torch.normal(0, 0.0003, kspace.shape).to(device)
        fname = os.path.split(tsmi_path)[-1]
        fname = fname.replace('tsmi_ref_', '')
        output_file = os.path.join(kspace_output_dir, f'ksp_{fname}')
        np.savez(output_file, y=kspace.to('cpu').numpy())
        print('Kspace saved to', output_file)

def get_tsmis_adjoint_svd_from_kspace(kspace_input_dir, adjoint_output_dir, device, nufft_operator, cut=0):
    kspace_paths = glob.glob(os.path.join(kspace_input_dir, f'*_cut{cut}.npz'))
    os.makedirs(adjoint_output_dir, exist_ok=True)
    for kspace_path in kspace_paths:
        # input_dir should have the full path to data/mrfscans/FISP_3T/ksp
        print(f'Processing {kspace_path}')
        y = torch.from_numpy(np.load(kspace_path)['y']).to(device)
        tsmi_adjoint = nufft_operator.adj(y, sens=None)
        print(tsmi_adjoint.shape)
        fname = os.path.split(kspace_path)[-1]
        output_file = os.path.join(adjoint_output_dir, fname.replace('ksp_', 'adjoint_'))
        np.savez(output_file, tsmi_adjoint=tsmi_adjoint.cpu().numpy())

if __name__ == '__main__':
    gpu_device = 0
    device = 'cuda:%s' % gpu_device if torch.cuda.is_available() else 'cpu'

    # =============================================================== Getting Reference TSMI for DELI-CS
    fisp_cut = 3
    k_svd = 5

    dict_path = f'./mrf_processing/SVD_dict_FISP_cut{fisp_cut}.mat'
    qmaps_output_dir = r'./datasets/Deli-CS/qmaps'
    tsmi_output_dir = r'./datasets/Deli-CS/reference_tsmi'
    kspace_output_dir = r'./datasets/Deli-CS/synthesized_ksp'
    adjoint_output_dir = r'./datasets/Deli-CS/adjoint_tsmi'
    qmaps_dir = r'' # Specify path to Deli-CS data!!!!
    # get_tsmis_reference_from_qmaps(dict_path, qmaps_dir, tsmi_output_dir, qmaps_output_dir, k_svd=k_svd, fisp_cut=fisp_cut,
    #                     mask_threshold=0.006)

    acquisition = 'fisp'
    ktraj, dcf, basis, time_frames, num_samples, im_shape, dictionary_path = load_traj_basis_dcf_2dscans(acquisition,
                                                                                                         cut=fisp_cut)
    ktraj = torch.tensor(ktraj, dtype=torch.float32)
    dcf = torch.tensor(dcf, dtype=torch.float32)
    basis = torch.tensor(basis[:5], dtype=torch.float32)
    NT = time_frames

    print(f"Shapes-- ktraj:{ktraj.shape}, dcf:{dcf.shape}, basis:{basis.shape}, timeframes: {NT}")
    print("dict path:", dictionary_path)

    lib_nufft = 'tkbn'  # 'tfin' or 'tkbn'
    NUFFT = NUFFT_Pytorch(im_shape, ktraj, basis, dcf=dcf, device=device, lib_nufft=lib_nufft)

    get_kspace_from_reference_tsmi(tsmi_dir=tsmi_output_dir,
                                   kspace_output_dir=kspace_output_dir,
                                   cut=fisp_cut,
                                   device=device,
                                   nufft_operator=NUFFT)

    get_tsmis_adjoint_svd_from_kspace(kspace_input_dir=kspace_output_dir,
                                      adjoint_output_dir=adjoint_output_dir,
                                      device=device,
                                      nufft_operator=NUFFT,
                                      cut=fisp_cut)