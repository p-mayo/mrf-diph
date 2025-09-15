import numpy as np
import torch
import scipy.io as sio


class group_dict_match:
    def __init__(self, param, mr_signal, device='cpu'):
        """
            This class sets up dictionary matching as follows:
            1) Load SVD MRF dictionary.
            2) Builds a MRIsynth dictionary.
            3) As an output, builds the function prox.dm, a joint MRF-MRIsynth dictionary matching, based steps 1&2.

            Adated from paper:
            Cauley SF, et al. Fast group matching for MR fingerprinting reconstruction. Magn Reson Med.
            2015 Aug;74(2):523-8. doi: 10.1002/mrm.25439. Epub 2014 Aug 28. PMID: 25168690; PMCID: PMC4700821.

            implementation:
            (c) Mohammad Golbabaee (m.golbabaee@bristol.ac.uk), 14.06.2024
            """

        # Load SVD MRF dictionary
        with torch.no_grad():
            dict_data = sio.loadmat(param['dir']['dict'])['dict']
            dict = {'D': torch.tensor(dict_data['D'][0, 0], dtype=torch.float32),
                    'V': torch.tensor(dict_data['V'][0, 0], dtype=torch.float32),
                    'lut': torch.tensor(dict_data['lut'][0, 0], dtype=torch.float32),
                    'normD': torch.tensor(dict_data['normD'][0, 0], dtype=torch.float32)
                    }

            del dict_data
            print('MRF dictionary loaded.')

            dict['D'] = dict['D'][:, :param['dim']['tsmi_shape'][0]]
            self.Dm = (dict['D'] * dict['normD']).to(device)  # un-normalised svd-mrf dict

            # Set MRI dictionary
            lut_ones = torch.cat((dict['lut'], torch.ones_like(dict['lut'])), dim=1)
            lut_ones[:, -1] = 0
            self.Dw = mr_signal(lut_ones).to(device)
            self.lut = dict['lut'].to(device)

            self.ngroups = param['gdm']['ngroups']
            self.corr_cutoff = param['gdm']['corr_cutoff']
            self.nbatch = param['gdm']['nbatch']
            self.device = device

            del dict

    def gdm_construct_and_apply(self, xm, xw, la):
        '''''
        # TSMI: xm.shape = [N (voxels), t (svd channels)] dtype = torch.complex64
        # multimodal MRI: xw.shape = [N (voxels), s (num mri modalities)] dtype = torch.float32 or torch.complex64

        TSMI: xm.shape = [t (svd channels), N (voxels)] dtype = torch.complex64
        Multimodal MRI: xw.shape = [s (num mri modalities), N (voxels)] dtype = torch.float32 or torch.complex64
        '''''
        with torch.no_grad():
            s = self.Dw.size(1)
            la = la.to(self.device, dtype=torch.float32)

            dict = {'D': torch.cat((self.Dm, self.Dw * torch.sqrt(la)), dim=1),
                    'lut': self.lut}
            dict['normD'] = torch.sqrt(torch.sum(torch.abs(dict['D']) ** 2, dim=1)).unsqueeze(1)
            dict['D'] /= dict['normD']
            dict = gdm_construct_dict(dict, s, self.ngroups)  # construct group dictionary

            if xw is None:
                # xw = torch.zeros(xm.size(0), s, dtype=torch.float32, device= xm.device)
                xw = torch.zeros(s, xm.size(1), dtype=torch.float32, device=xm.device)

            xw = torch.real(xw)
            xw[xw<0]=0 #1/5/24 to ensure real and non-negative (tho should also work without)

            # run group dictionary matching (search)
            # q, pd, ind = gdm_apply(dict, torch.cat((xm, xw * torch.sqrt(la)), dim=1), s,
            #                           cutoff= self.corr_cutoff, nbatch = self.nbatch)
            q, pd, ind = gdm_apply(dict, torch.cat((xm.t(), xw.t() * torch.sqrt(la)), dim=1), s,
                                   cutoff=self.corr_cutoff, nbatch=self.nbatch)

            xm = self.Dm[ind] * pd.unsqueeze(1)
            xw = self.Dw[ind] * torch.abs(pd).unsqueeze(1)
            q = torch.cat((q, torch.view_as_real(pd)), dim=1)

            return q.t(), xm.t(), xw.t(), ind.t()


def gdm_apply(g_dict, x, s_, cutoff=.99, nbatch=10):
    # print('group dictionary matching ...')
    # torch.cuda.empty_cache()
    gdm_device = x.device

    with torch.no_grad():

        ## Get variables
        Q = g_dict['lut'].size(1)
        N, T = x.shape
        blockSize = N // nbatch
        # blockSize = int(torch.ceil(torch.tensor(N / nbatch)))

        ## Group Matching Fit
        g_means = g_dict['group_means']
        g_means = g_means / torch.sqrt(torch.sum(torch.abs(g_means) ** 2, dim=1, keepdim=True))
        n_groups = torch.unique(g_dict['group_labels']).size(0)
        g_means = g_means[torch.unique(g_dict['group_labels']).to(torch.int64), :] #added jul24

        # correlate signals with means of groups
        x_groups_mask = torch.zeros((n_groups, N), dtype=bool, device=gdm_device)
        for i in range(nbatch):
            if i < nbatch - 1:
                cind = torch.arange(i * blockSize, (i + 1) * blockSize, device=gdm_device)
            else:
                cind = torch.arange(i * blockSize, N, device=gdm_device)

            corr_x_gmeans = torch.abs(torch.matmul(g_means[:, :-s_].to(torch.cfloat), x[cind, :-s_].conj().t())) + \
                            torch.matmul(g_means[:, -s_:], torch.real(x[cind, -s_:]).t())
            # prune groups by a cutoff
            corr_cutoff = cutoff * torch.max(corr_x_gmeans, dim=0)[0]  # gives cutoff for each voxel
            x_groups_mask[:, cind] = corr_x_gmeans >= corr_cutoff

        del corr_cutoff, corr_x_gmeans, cind
        # torch.cuda.empty_cache()

        # find match within (un-pruned) group
        q = torch.zeros(N, Q, dtype=torch.float32, device=gdm_device)  # qmaps without pd
        ind = torch.zeros(N, dtype=torch.int64, device=gdm_device)
        pd = torch.zeros(N, dtype=torch.complex64, device=gdm_device)

        for i in range(nbatch):  # Batch the image (voxels) for Dictionary Matching to avoid memory overflow
            if i < nbatch - 1:
                cind = torch.arange(i * blockSize, (i + 1) * blockSize, device=gdm_device)
                cv = torch.zeros(blockSize, n_groups, dtype=torch.float32, device=gdm_device)
                ci = torch.zeros(blockSize, n_groups, dtype=torch.int64, device=gdm_device)
            else:
                cind = torch.arange(i * blockSize, N, device=gdm_device)
                cv = torch.zeros(len(cind), n_groups, dtype=torch.float32, device=gdm_device)
                ci = torch.zeros(len(cind), n_groups, dtype=torch.int64, device=gdm_device)

            x_b = x[cind, :]
            x_groups_mask_b = x_groups_mask[:, cind]

            for k in range(n_groups):
                mask = x_groups_mask_b[k]
                voxels_k = x_b[mask]

                group_k_idx = (g_dict['group_labels'][:, 0] == k).nonzero()[:, 0].to(torch.int64)

                corr_x_group_k = torch.abs(
                    torch.mm(g_dict['D'][group_k_idx, :-s_].to(torch.cfloat), voxels_k[:, :-s_].conj().t())) + \
                                 torch.abs(torch.mm(g_dict['D'][group_k_idx, -s_:].to(torch.cfloat),
                                                    voxels_k[:, -s_:].conj().t()))

                cv[mask, k], idx_tmp = torch.max(corr_x_group_k, dim=0)
                ci[mask, k] = group_k_idx[idx_tmp.to(torch.int64)]

            pd_b, idx_groups = torch.max(cv,
                                         dim=1)  # highest correlation of an atom among all (un-pruned) groups for a voxel.
            ind_b = ci[(torch.arange(ci.shape[0]), idx_groups)]  # index of matchest atom within matchest group

            pd_b = pd_b / g_dict['normD'][ind_b].squeeze(1)  # magnitude of pd
            pd_phase = (x_b[:, :-s_] * g_dict['D'][ind_b, :-s_].conj()).sum(dim=1)
            pd_phase = pd_phase / torch.abs(pd_phase)
            pd_b = pd_b * pd_phase  # pd phase added

            q[cind] = g_dict['lut'][ind_b]
            pd[cind] = pd_b
            ind[cind] = ind_b

            del ci, cv, corr_x_group_k, mask, voxels_k, x_b, x_groups_mask_b, pd_b, pd_phase, ind_b, idx_groups
            # torch.cuda.empty_cache()

        return q, pd, ind


def gdm_construct_dict(dict, s, n_groups):
    with torch.no_grad():
        N, T = dict['D'].shape
        device = dict['D'].device
        n_signals_per_group = int(torch.ceil(torch.tensor(N / n_groups)))
        dict['group_means'] = torch.zeros(n_groups, T, dtype=torch.float32).to(device)
        dict['group_labels'] = torch.zeros(N, 1, dtype=torch.int16).to(device)
        remain_idxs = torch.arange(N).to(device)

        for ii in range(n_groups):
            # select a random signal from remaining signals
            n_remain_idxs = len(remain_idxs)
            ind = torch.randint(n_remain_idxs, (1,))
            s_0 = dict['D'][ind]

            # Correlate with ungrouped elements
            signals_dp = torch.abs(torch.mm(dict['D'][remain_idxs, :-s], s_0[:, :-s].conj().t())) + \
                         torch.mm(dict['D'][remain_idxs, -s:], s_0[:, -s:].conj().t())

            # assign best n_signals_per_group elements to new group
            _, max_dps = torch.sort(signals_dp, dim=0, descending=True)
            max_dps = max_dps[:min(n_signals_per_group, n_remain_idxs)]
            new_group = remain_idxs[max_dps]
            dict['group_labels'][new_group] = ii

            # Form group mean signal s_k
            dict['group_means'][ii] = torch.mean(dict['D'][new_group], dim=0)

            # Update remaining dict
            remain_idxs = remain_idxs[torch.logical_not(torch.isin(remain_idxs, new_group))]
            if len(remain_idxs) == 0:
                break

        return dict

