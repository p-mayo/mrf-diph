# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 21:12:41 2024

@author: pm15334
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from guided_diffusion.mrf_image_datasets_2d import  ref_factors

vmax = [2.5, .25]

def get_nrmse(ref, test):
    return torch.norm(ref-test)/torch.norm(ref)

def get_mape(ref, test, mask = None):
    error = torch.abs(ref-test)
    if mask is not None:
        valid_idxs = torch.nonzero(ref*mask, as_tuple=True)
    else:
        valid_idxs = torch.nonzero(torch.ones_like(ref), as_tuple = True)
    return torch.mean(error[valid_idxs]/torch.abs(ref[valid_idxs]))

def save_pngs(qmaps, tsmis, ref_qmaps, ref_tsmis, mask, output_path):
    fig, axs = plt.subplots(1, 7, figsize=(20, 5))
    for p in range(7):
        if p <2:
            im = axs[p].imshow(qmaps[p], cmap='afmhot', vmax=vmax[p], vmin=0)
            error = get_mape(ref_qmaps[p], qmaps[p], mask)
            axs[p].set_title(f'T{p+1}\nMAPE = {100*error:0.2f}')
        else:
            im = axs[p].imshow(torch.abs(tsmis[p-2]), cmap='gray', vmax=torch.abs(ref_tsmis[p-2]).max(), vmin=0)
            error = get_nrmse(ref_tsmis[p-2], tsmis[p-2])
            axs[p].set_title(f'Coefficient {p - 1}\nNRMSE = {100*error:0.2f}')
        divider = make_axes_locatable(axs[p])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        axs[p].set_xticks([])
        axs[p].set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    qmaps_ref_dir = './datasets/Deli-CS/qmaps'
    tsmis_ref_dir = './datasets/Deli-CS/reference_tsmi'
    models_path = r'./models'
    output_path = './evaluations'
    modes = ['conditional', 'unconditional']
    checkpoint = 100000
    image_size = 224
    n_samples = 3
    vol = 0
    slices = [5,10,15]
    cut = 3
    slce_to_show = 5
    lambdas = ['0.1', '0.01', '0.001', '0.0001', '0.00001', '0.000001', '0.0000001', '0.00000001', '0.000000001', '0.0']
    fig_recons, axs_recons = plt.subplots(1, 4, figsize=(20, 4))
    for mode in modes:
        all_results = None

        mode_path = os.path.join(models_path, f'cut_{cut}_imgsize_{image_size}_{mode}')
        sampling_experiments = os.listdir(mode_path)
        for lambda_value in lambdas:
            experiment = f'mrfdiph_samples_lambda{lambda_value}'
            setting = experiment.split('_')[-1]
            experiment_path = os.path.join(mode_path, experiment)
            samples_path = os.path.join(experiment_path, f'samples_ema_0.9999_{checkpoint}_{n_samples}x10x{image_size}x{image_size}.npz')
            if not os.path.exists(samples_path):
                continue

            # To check error logs
            csv_path = os.path.join(experiment_path, 'log_res.csv')
            results = pd.read_csv(csv_path)
            if lambda_value == '0.0':
                results['lmbda'] = 1e-10
            if all_results is None:
                all_results = results
            else:
                all_results = pd.concat([all_results, results], ignore_index=True, sort=False)

            exp_results = np.load(samples_path)

            qmaps = torch.from_numpy(exp_results['qmaps'])
            tsmis = torch.from_numpy(exp_results['z_t'])*ref_factors[cut]

            for idx, slce in enumerate(slices):
                qmaps_ref_path = os.path.join(qmaps_ref_dir, f'qmaps_vol{vol}_slice{slce}.npz')
                ref_qmaps = torch.from_numpy(np.load(qmaps_ref_path)['qmaps'])

                mask_path = os.path.join(qmaps_ref_dir, f'mask_vol{vol}_slice{slce}_th0.006.npz')
                ref_mask = torch.from_numpy(np.load(mask_path)['mask'])

                tsmi_path = os.path.join(tsmis_ref_dir, f'tsmi_ref_vol{vol}_slice{slce}_cut{cut}.npz')
                ref_tsmi = torch.from_numpy(np.load(tsmi_path)['tsmi_ref'])

                img_output_path = os.path.join(output_path, f'vol{vol}.{slce}_{mode}_{setting}.png')
                save_pngs(qmaps[idx], tsmis[idx], ref_qmaps, ref_tsmi[0], ref_mask, img_output_path)

        valid_idxs = (all_results['slice'] == slce_to_show) & (all_results['iteration'] == 1)
        axs_recons[0].plot(all_results.loc[valid_idxs]['lmbda'], all_results.loc[valid_idxs]['mape_t1'],
                           label=mode.title())
        axs_recons[1].plot(all_results.loc[valid_idxs]['lmbda'], all_results.loc[valid_idxs]['mape_t2'],
                           label=mode.title())
        axs_recons[2].plot(all_results.loc[valid_idxs]['lmbda'], all_results.loc[valid_idxs]['nrmse_zt'],
                           label=mode.title())
        axs_recons[3].plot(all_results.loc[valid_idxs]['lmbda'], all_results.loc[valid_idxs]['nrmse_kspace_zt'],
                           label=mode.title())
    axs_recons[0].set_title('MAPE T1')
    axs_recons[1].set_title('MAPE T2')
    axs_recons[2].set_title('NRMSE TSMI')
    axs_recons[3].set_title('NRMSE k-space')

    for ax in axs_recons.ravel():
        ax.legend()
        ax.set_xscale('log')
    fig_recons.savefig(os.path.join(output_path, f'lambda_vs_error_{vol}.{slce_to_show}.png'))