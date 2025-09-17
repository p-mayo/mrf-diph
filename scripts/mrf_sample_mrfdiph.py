"""
Script to sample using pre-trained diffusion model. This code has been adapted from original source
https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py

@author: PM
"""
import os
import gc
import torch
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.mrf_image_datasets_2d import load_data, ref_factors

from guided_diffusion.script_util import (
    mrf_create_model_and_diffusion,
    mrf_model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)

from mrf_processing.utils_dm_exact import dict_match
from mrf_processing.utils_acquisition2d import NUFFT_Pytorch, load_traj_basis_dcf_2dscans
from evaluations.experimental_results import get_nrmse, get_mape

def save_metrics(file_path, results, all_rows, iterations):
    with open(file_path, 'w') as f:
        keys = results.keys()
        f.write('iteration,' + ','.join(keys) +'\n')
        it = iterations
        for i in range(all_rows):
            str_res = f'{it},'
            for k in keys:
                str_res += f'{results[k][i]},'
            f.write(str_res[:-1] + '\n')
            it = it - 1 if it > 1 else iterations

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    logger.log('Parameters:')
    params_str = ''
    for k, v in args_to_dict(args, mrf_model_and_diffusion_defaults().keys()).items():
        params_str += f'\t{k.upper()}: {v}\n'
    logger.log(params_str)
    logger.log('creating model and diffusion...')
    model, diffusion = mrf_create_model_and_diffusion(
        **args_to_dict(args, mrf_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log('creating dataset...')
    test_vols = [0, 11] if args.test_vols == '-1' else [int(x) for x in args.test_vols.split(',')]
    test_slices = range(30) if args.test_slices == '-1' else [int(x) for x in args.test_slices.split(',')]
    logger.log(f'test_vols: {test_vols}')
    logger.log(f'test_slices: {test_slices}')
    data = load_data(
        image_size=args.image_size,
        input_dir=args.input_dir,
        ref_dir=args.ref_dir,
        batch_size=args.batch_size,
        vols=test_vols,
        slices=test_slices,
        deterministic=True,
        augment=False,
        cut=args.cut,
        use_condition=args.use_condition, # To retrieve AHy for the proxop
    )

    ##################################################################
    # initialise proxp, forward and adjoint operators
    # for a given MRF dictionary & acquisition trajectory
    ##################################################################
    logger.log("initializing proximal, forward and adjoint operators")
    acquisition='fisp'
    ktraj, dcf, basis, time_frames, num_samples, im_shape, dictionary_path = load_traj_basis_dcf_2dscans(acquisition,
                                                                                                     cut=args.cut)
    ktraj = torch.tensor(ktraj, dtype=torch.float32)
    dcf = torch.tensor(dcf, dtype=torch.float32)
    basis = torch.tensor(basis[:5], dtype=torch.float32)
    NT = time_frames

    logger.log(f'Shapes-- ktraj:{ktraj.shape}, dcf:{dcf.shape}, basis:{basis.shape}, timeframes: {NT}')
    logger.log(f'dict path: {dictionary_path}')

    lib_nufft = 'tkbn'  # 'tfin' or 'tkbn'

    NUFFT = NUFFT_Pytorch(im_shape, ktraj, basis, dcf=dcf, device=dist_util.dev(), lib_nufft=lib_nufft)
    proxf = NUFFT.proxf

    sample_fn = (
        diffusion.p_sample if not args.use_ddim else diffusion.ddim_sample
    )

    # Variables for ADMM
    logger.log("setting variables for ADMM ...")
    xi = args.xi
    lmbda = args.lmbda

    if args.sigma_level == -1:
        sqrt_alpha_bar = diffusion.sqrt_alphas_cumprod
        sqrt_one_minus_alpha_bar = diffusion.sqrt_one_minus_alphas_cumprod
        indices = list(range(diffusion.num_timesteps))[::-1]
    else:
        t_idx = 27
        timesteps = 30
        sqrt_alpha_bar = torch.tensor([diffusion.sqrt_alphas_cumprod[t_idx]]*timesteps)
        sqrt_one_minus_alpha_bar = torch.tensor([diffusion.sqrt_one_minus_alphas_cumprod[t_idx]]*timesteps)
        indices = list(range(timesteps))[::-1]

    sigmas = sqrt_one_minus_alpha_bar / sqrt_alpha_bar
    mus = lmbda / (sigmas ** 2)

    if args.gamma_fixed:
        gamma = [args.gamma] * len(mus)
    else:
        gamma = args.gamma * mus
    logger.log('xi:', xi)
    logger.log('gamma:', gamma)
    logger.log('lambda:', lmbda)

    # Setting up Dictionary Matching (DM)
    logger.log("setting up dictionary matching (DM)...")
    param = {'dir':{'dict':dictionary_path},
             'dim':{'tsmi_shape': (5,230,230)},
             'gdm':{'ngroups':100, 'corr_cutoff':0.99, 'nbatch':10}}

    DM = dict_match(param=param, device=dist_util.dev())
    dm = DM.dm_apply

    if args.report:
        all_vols = []
        all_slices = []
        all_gammas = []
        all_lmbdas = []
        all_xis = []

        all_nrmses_zt = []
        all_nrmses_xtilda = []
        all_nrmses_xhat = []
        all_nrmses_kspace_zt = []
        all_nrmses_kspace_xtilda = []
        all_nrmses_kspace_xhat = []
        all_mapes_t1 = []
        all_mapes_t2 = []

    dpm_times =[]
    prox_times =[]
    dm_times = []
    iter_times = []

    all_xts = []
    all_xtildas = []
    all_xhats = []
    all_zs = []
    all_qmaps = []

    logger.log("sampling...")

    with torch.no_grad():
        for vol in test_vols:
            for slce in test_slices:
                logger.log(f'Volume: {vol}, slice: {slce}')
                kspace_path = os.path.join(args.kspace_dir, f'ksp_vol{vol}_slice{slce}_cut{args.cut}.npz')
                y_slice = torch.from_numpy(np.load(kspace_path)['y'])
                y_slice = y_slice.to(dist_util.dev())/ref_factors[args.cut]
                sens = None

                qmaps_ref_path = os.path.join(args.qmaps_ref_dir, f'qmaps_vol{vol}_slice{slce}.npz')
                ref_qmaps = np.load(qmaps_ref_path)['qmaps']
                ref_qmaps = torch.from_numpy(ref_qmaps)

                mask_path = os.path.join(args.qmaps_ref_dir, f'mask_vol{vol}_slice{slce}_th0.006.npz')
                ref_mask = np.load(mask_path)['mask']
                ref_mask = torch.from_numpy(ref_mask)

                # ==============================================================
                # Compute AHy (SVDMRF Recon)
                # ==============================================================
                AHy =  NUFFT.adj(y_slice, sens)

                # ==============================================================
                # Getting data from Data Loader
                # ==============================================================
                gt, model_kwargs = next(data)

                bs, c, h, w = gt.shape
                shape = gt.shape
                x_t = torch.randn(*shape).to(dist_util.dev())      # Initialise x_T as pure Gaussian noise

                if args.use_condition:
                    model_kwargs['low_res'] = model_kwargs['low_res'].to(dist_util.dev())
                gt = torch.nn.functional.pad(gt, pad=(3, 3, 3, 3), mode='constant', value=0)
                gt = gt.reshape(bs, c // 2, 2, 230, 230).to(dist_util.dev())
                gt = gt.permute(0, 1, 3, 4, 2).contiguous()
                gt = gt[:, :, :, :, 0] + 1j * gt[:, :, :, :, 1]

                # Setting variables for ADMM
                v_t = torch.zeros_like(AHy)
                z_t = torch.zeros_like(AHy)

                logger.log('\tstarting sampling\n\t')
                start_time = datetime.datetime.now()
                for i in indices:
                    start_iter = datetime.datetime.now()
                    # ====================================================================
                    #  G E T T I N G   X_0_TILDA AT T I M E S T E P  'T'
                    # ====================================================================
                    if args.sigma_level ==-1:
                        t = torch.tensor([i])
                    else:
                        t = torch.tensor([t_idx])
                    start_dpm = datetime.datetime.now()
                    x_0_tilda = sample_fn(
                        model,
                        x_t.to(dist_util.dev()),
                        t.to(dist_util.dev()),
                        clip_denoised=args.clip_denoised,
                        model_kwargs=model_kwargs if args.use_condition else {},
                    )['pred_xstart']

                    end_dpm = datetime.datetime.now()
                    # Need to format x_tilda to deal with complex
                    x_0_tilda = x_0_tilda.reshape(bs, c // 2, 2, args.image_size, args.image_size)  # .to(dist_util.dev())
                    x_0_tilda = torch.nn.functional.pad(x_0_tilda, pad=(3, 3, 3, 3), mode='constant', value=0)
                    x_0_tilda = x_0_tilda.permute(0, 1, 3, 4, 2).contiguous()
                    x_0_tilda = x_0_tilda[:, :, :, :, 0] + 1j * x_0_tilda[:, :, :, :, 1]

                    # ==========================================================================
                    # G E T T I N G   X_HAT_0   V I A   P R O X O P
                    # ==========================================================================
                    if args.proximal:
                        x = (mus[i] * x_0_tilda + gamma[i] * (z_t - v_t/gamma[i])) / (mus[i] + gamma[i])
                        x_init = torch.zeros_like(x) if (i==len(indices) - 1) else x_0_hat
                        proxf_param = {                     # hyperparams of proxf that need tuning
                            'rho' : mus[i] + gamma[i] ,     # important data fidelity tradeoff param
                            'cg_maxiter' : args.proxop_cg_maxiter,  # 10 or less iters could be enough (see DOLCE)-- early-stop
                            'cg_tol' : args.proxop_cg_tol,  # similarly tol could be higher e.g. 1e-3 to early stop.
                            'x0' : x_init.clone(),       # NB: for iterative usage of prox, use x0 (initialisation)
                                                            # from previous iter of x to accelerate
                        }
                        start_proximal = datetime.datetime.now()
                        x_0_hat = proxf(x, AHy.to(dist_util.dev()), sens, proxf_param)
                        end_proximal = datetime.datetime.now()
                    else:
                        x_0_hat = x_0_tilda.clone()
                    torch.cuda.empty_cache()
                    gc.collect()

                    # ==========================================================================
                    # Z_{T-1}   U P D A T E   V I A   D M
                    # ==========================================================================
                    if args.dict_match:
                        tsmi = x_0_hat + v_t/gamma[i]
                        start_dm = datetime.datetime.now()
                        q_hat, z_t, _, _ = dm(xm=tsmi[0].reshape(5,-1), xw=None, la=torch.tensor([0]))
                        z_t = z_t.reshape(tsmi.shape)
                        q_hat = q_hat.reshape([4, 230, 230])
                        end_dm = datetime.datetime.now()
                    else:
                        tsmi = x_0_hat.clone()
                        if args.report:
                            q_hat, _, _, _ = dm(xm=tsmi[0].reshape(5, -1), xw=None, la=torch.tensor([0]))
                            q_hat = q_hat.reshape([4, 230, 230])

                    # ==========================================================================
                    # DUAL VARIABLE UPDATE
                    # ==========================================================================
                    v_t = v_t + gamma[i]*(x_0_hat - z_t)

                    # ==========================================================================
                    # ADDING NOISE
                    # ==========================================================================
                    # Need now to revert to have them all as real
                    if args.dict_match:
                        x_0_hat_t = torch.view_as_real(z_t)
                    else:
                        x_0_hat_t = torch.view_as_real(x_0_hat)
                    x_0_hat_t = x_0_hat_t.permute(0, 1, 4, 2, 3).contiguous()
                    x_0_hat_t = x_0_hat_t.reshape(bs, c, 230, 230)
                    x_0_hat_t = x_0_hat_t[:,:,3:-3, 3:-3]

                    epsilon_hat_t = (1/sqrt_one_minus_alpha_bar[i])*(x_t - sqrt_alpha_bar[i]*x_0_hat_t)
                    epsilon = torch.randn_like(epsilon_hat_t)
                    if i > 0:
                        x_t = sqrt_alpha_bar[i-1]*x_0_hat_t + sqrt_one_minus_alpha_bar[i-1]*((xi**0.5)*epsilon + ((1-xi)**0.5)*epsilon_hat_t)
                    else:
                        # Noise is not added at the last iteration
                        x_t = x_0_hat_t

                    end_iter = datetime.datetime.now()
                    iter_times.append(end_iter - start_iter)
                    dpm_times.append(end_dpm - start_dpm)
                    prox_times.append(end_proximal - start_proximal)
                    dm_times.append(end_dm - start_dm)
                    if args.report:
                        # Getting metrics for reporting
                        mape_t1 = get_mape(ref_qmaps[0].cpu(), q_hat[0].cpu(), ref_mask.cpu())
                        mape_t2 = get_mape(ref_qmaps[1].cpu(), q_hat[1].cpu(), ref_mask.cpu())

                        nrmse_z = get_nrmse(gt.cpu(), z_t.cpu())
                        nrmse_x_tilda = get_nrmse(gt.cpu(), x_0_tilda.cpu())
                        nrmse_x_hat = get_nrmse(gt.cpu(), x_0_hat.cpu())
                        nrmse_kspace_xhat = get_nrmse(y_slice, NUFFT.fwd(x_0_hat.to(dist_util.dev()), sens))
                        nrmse_kspace_zt = get_nrmse(y_slice, NUFFT.fwd(z_t.to(dist_util.dev()), sens))
                        nrmse_kspace_xtilda = get_nrmse(y_slice, NUFFT.fwd(x_0_tilda.to(dist_util.dev()), sens))

                        logger.log(f'Iteration {i} - NRMSEs\tZ: {nrmse_z}\tX Tilda: {nrmse_x_tilda}\tX Hat: {nrmse_x_hat}'
                                   f'\tKspace Xhat: {nrmse_kspace_xhat}\tKSpace Z_T: {nrmse_kspace_zt}',
                                   f'\tMAPE T1: {mape_t1}\tMAPE T2: {mape_t2}')

                        all_vols.append(vol)
                        all_slices.append(slce)
                        all_gammas.append(gamma[i])
                        all_lmbdas.append(lmbda)
                        all_xis.append(xi)

                        all_nrmses_zt.append(nrmse_z)
                        all_nrmses_xtilda.append(nrmse_x_tilda)
                        all_nrmses_xhat.append(nrmse_x_hat)
                        all_nrmses_kspace_zt.append(nrmse_kspace_zt)
                        all_nrmses_kspace_xtilda.append(nrmse_kspace_xtilda)
                        all_nrmses_kspace_xhat.append(nrmse_kspace_xhat)

                        all_mapes_t1.append(mape_t1)
                        all_mapes_t2.append(mape_t2)
                        torch.cuda.empty_cache()
                        gc.collect()

                        titles = [r'$\tilde{x}_0$', r'$\hat{x}_0$', r'$z_t$', 'reference']
                        fig, axs = plt.subplots(4, 5, figsize=(25, 20))
                        for reci, rec in enumerate([x_0_tilda, x_0_hat, z_t, gt]):
                            axs[reci, 0].set_ylabel(titles[reci])
                            for col in range(5):
                                im = axs[reci, col].imshow(torch.abs(rec[0,col]).cpu(), vmax = torch.abs(gt[0, col]).cpu().max(), vmin = 0)
                                fig.colorbar(im, ax=axs[reci, col])
                                axs[reci, col].set_xticks([])
                                axs[reci, col].set_yticks([])
                        plt.tight_layout()
                        plt.subplots_adjust(wspace=0, hspace=0)
                        plt.savefig(os.path.join(logger.get_dir(), f'vol{vol}.{slce}_iter_{i}.png'))
                        plt.close()
                end_time = datetime.datetime.now()
                total_time = end_time - start_time
                logger.log(f'\tsampling finished, total time: {total_time}')
                logger.log('Total Iteration time:', np.sum(iter_times))
                step_times = np.sum(dpm_times) + np.sum(prox_times) + np.sum(dm_times)
                logger.log('Total dpm time:', np.sum(dpm_times), f'{np.sum(dpm_times)*100/step_times:.2f}')
                logger.log('Total prox time:', np.sum(prox_times), f'{np.sum(prox_times)*100/step_times:.2f}')
                logger.log('Total dm time:', np.sum(dm_times), f'{np.sum(dm_times)*100/step_times:.2f}')

                all_zs.extend([z_t.cpu().numpy()])
                all_xts.extend([x_t.cpu().numpy()])
                all_xtildas.extend([x_0_tilda.cpu().numpy()])
                all_xhats.extend([x_0_hat.cpu().numpy()])
                all_qmaps.extend([q_hat.unsqueeze(0).cpu().numpy()])
                logger.log(f"created {len(all_xts) * args.batch_size} samples")

    if args.report:
        dict_res = {'vol': all_vols,
                    'slice': all_slices,
                    'gamma': all_gammas,
                    'lmbda': all_lmbdas,
                    'xi': all_xis,
                    'mape_t1': all_mapes_t1,
                    'mape_t2': all_mapes_t2,
                    'nrmse_zt': all_nrmses_zt,
                    'nrmse_xtilda': all_nrmses_xtilda,
                    'nrmse_xhat': all_nrmses_xhat,
                    'nrmse_kspace_zt': all_nrmses_kspace_zt,
                    'nrmse_kspace_xtilda': all_nrmses_kspace_xtilda,
                    'nrmse_kspace_xhat': all_nrmses_kspace_xhat,
                    }

        save_metrics(os.path.join(logger.get_dir(), 'log_res.csv'), dict_res,
                     diffusion.num_timesteps * len(test_vols) * len(test_slices),
                     diffusion.num_timesteps)
    x_t_arr = np.concatenate(all_xts, axis=0)
    zs_arr = np.concatenate(all_zs, axis=0)
    xtilda_arr = np.concatenate(all_xtildas, axis=0)
    xhat_arr = np.concatenate(all_xhats, axis=0)
    qmaps_arr = np.concatenate(all_qmaps, axis=0)
    shape_str = 'x'.join([str(x) for x in x_t_arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{args.model_path.split('/')[-1][:-3]}_{shape_str}.npz")
    logger.log(f'saving to {out_path}')
    np.savez(out_path, x_t=x_t_arr, z_t=zs_arr, x_tilda=xtilda_arr, x_hat=xhat_arr, qmaps=qmaps_arr)
    logger.log('sampling complete')

def create_argparser():
    defaults = dict(
        input_dir='',
        ref_dir='',
        qmaps_ref_dir='',
        kspace_dir='',
        clip_denoised=False,
        batch_size=1,
        use_ddim=False,
        model_path='',
        ksvd=5,
        cut=3,
        test_vols='0,11',
        test_slices='-1',
        lmbda=0.0001,
        gamma=1e-4,
        xi=1.,
        gamma_fixed=False,
        proxop_cg_maxiter=5,
        proxop_cg_tol=1e-5,
        dict_match=True,
        proximal=True,
        sigma_level=-1.,
        report=True,
    )
    defaults.update(mrf_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
