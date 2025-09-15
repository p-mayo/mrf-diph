# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:18:37 2024

@author: pm15334
"""
import os
import torch
import numpy as np

tsmi_paths = {'tsmi_ref' : r'/home/pm15334/data/mrfscans/3DTGAS/synthesized_tsmi/',
              'tsmi_adjoint' : r'/home/pm15334/data/mrfscans/3DTGAS/adjoint_tsmi/'
              }
use_mask = True
cut = 3

for key, tsmi_path in tsmi_paths.items():
    files = os.listdir(tsmi_path)

    mins = []
    maxs = []
    means = []
    stds = []
    all_tsmis = []
    tsmis_channel_0 = []
    tsmis_channel_1 = []
    tsmis_channel_2 = []
    tsmis_channel_3 = []
    tsmis_channel_4 = []
    for f in files:
        if f'cut{cut}' not in f:
            continue
        tsmi = os.path.join(tsmi_path, f)
        tsmi = torch.from_numpy(np.load(tsmi)[key])
        if tsmi.shape[-1] == 230:
            tsmi = tsmi[:,:,3:-3,3:-3]
        tsmi = torch.view_as_real(tsmi)

        all_tsmis.append(tsmi.squeeze().numpy())
        mins.append(tsmi.min())
        maxs.append(tsmi.max())
        means.append(tsmi.mean())
        stds.append(tsmi.std())

    all_tsmis = np.abs(np.stack(all_tsmis))
    print('All TSMIs shape', all_tsmis.shape)
    print(f'ALL - MIN: {min(mins):.4f}. MAX: {max(maxs):.4f}, MEAN: {np.mean(means):.4f}, STD: {np.mean(stds):.4f}')
    print(f'ALL - 95 Percentile {np.percentile(all_tsmis.ravel(), 95):.6f}, {np.percentile(all_tsmis.ravel(), 95):.3f}')
    print(f'ALL - 99 Percentile {np.percentile(all_tsmis.ravel(), 99):.6f}, {np.percentile(all_tsmis.ravel(), 99):.3f}')
    # all_tsmis = torch.stack(all_tsmis, dim=0) # Shape: 120, 5, 230, 230, 2


""" This outputs:
All TSMIs shape (360, 5, 256, 256, 2)
ALL - MIN: -0.0525. MAX: 0.0515, MEAN: -0.0000, STD: 0.0017
ALL - 95 Percentile 0.003997, 0.004
ALL - 99 Percentile 0.008240, 0.008

All TSMIs shape (360, 5, 224, 224, 2)
ALL - MIN: -0.0000. MAX: 0.0000, MEAN: -0.0000, STD: 0.0000
ALL - 95 Percentile 0.000004, 0.000
ALL - 99 Percentile 0.000007, 0.000


"""

# Checking reference and adjoint
tsmi_reference_path = r'/home/pm15334/data/mrfscans/3DTGAS/synthesized_tsmi'
tsmi_adjoint_path = '/home/pm15334/data/mrfscans/3DTGAS/adjoint_tsmi'
tsmi_png_path = '/home/pm15334/data/mrfscans/3DTGAS/tsmi_png'

os.makedirs(tsmi_png_path, exist_ok=True)

# Reference has file name format: tsmi_ref_vol{vol}_slice{slice}_cut{cut}.npz
# Adjoint has file name format: adjoint_tsmi_vol{vol}_slice{slice}_cut{cut}.npz

import glob
import matplotlib.pyplot as plt
files = glob.glob(os.path.join(tsmi_reference_path, f'*_cut{cut}.npz'))

for file in files:
    tsmi_ref = np.load(file)['tsmi_ref']
    fname = os.path.split(file)[-1]
    fname = fname.replace('tsmi_ref', 'adjoint_tsmi')
    tsmi_adj = np.load(os.path.join(tsmi_adjoint_path, fname))['tsmi_adjoint']

    fig, axs = plt.subplots(2, 10, figsize=(20, 5))
    for c in range(5):
        axs[0, c].imshow(np.real(tsmi_ref[0, c]))
        axs[0, 5 + c].imshow(np.imag(tsmi_ref[0, c]))

        axs[1, c].imshow(np.real(tsmi_adj[0, c]))
        axs[1, 5 + c].imshow(np.imag(tsmi_adj[0, c]))

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(os.path.join(tsmi_png_path, fname.replace('adjoint_', '').replace('.npz', '.png')))
    plt.close()
    break