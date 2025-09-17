import torch
import scipy.io as sio


class dict_match:
    def __init__(self, param, device='cpu'):
        """
            This class sets up dictionary matching as follows:
            1) Load SVD MRF dictionary.
            2) Builds a MRIsynth dictionary.
            3) As an output, builds the function prox.dm, a joint MRF-MRIsynth dictionary matching, based steps 1&2.
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
            self.Dnorm = torch.sqrt((torch.abs(self.Dm)**2).sum(dim=1))
            self.Dm /= self.Dnorm.unsqueeze(1)
            self.lut = dict['lut'].to(device)

            self.nblocks = param['gdm']['nbatch']
            self.device = device

            del dict

    def dm_apply(self, xm,**kwargs):
        '''''
        Performs (exact/brute-force) dictionary matching, given an input xm. 
        Input:
            xm: mrf image timeseries of shape = [t (svd channels), N (voxels)] dtype = torch.complex64
        Outputs: 
            q: quantitative maps q[:,:-2] & real & imag parts of proton density (q[:,-2:])
            x: dictionary-matched image timeseries of shape [t,N]
            ind: matched dictoinary indices per voxel, shape [N] 

        (c) Mohammad Golbabaee (m.golbabaee@bristol.ac.uk)
        '''''
        # print('group dictionary matching ...')
        # torch.cuda.empty_cache()

        with torch.no_grad():
            d,Q = self.lut.shape
            xm = xm.t()
            # x = x.detach().clone().t()
            N, T = xm.shape
            blockSize = N // self.nblocks

            q = torch.zeros(N, Q, dtype=torch.float32, device=self.device)
            ind = torch.zeros(N, dtype=torch.long, device=self.device)
            pd = torch.zeros(N, dtype=torch.complex64, device=self.device)

            for i in range(self.nblocks):
                if i < self.nblocks - 1:
                    cind = slice((i) * blockSize, (i + 1) * blockSize)
                else:
                    cind = slice(i * blockSize, N)

                xb = xm[cind, :]
                Nb = xb.shape[0]
                prod = torch.mm(xb , self.Dm.to(torch.cfloat).conj().t())

                _, ind[cind] = torch.max(torch.abs(prod), dim=1)
                q[cind, :] = self.lut[ind[cind]]
                pd[cind] = prod[torch.arange(Nb), ind[cind]]#/self.Dnorm[ind[cind]]

            del prod, xb, cind, xm
            # torch.cuda.empty_cache()

            x = self.Dm[ind] * pd.unsqueeze(1)
            pd /= self.Dnorm[ind]
            q = torch.cat((q, torch.view_as_real(pd)), dim=1)

            return q.t(), x.t(), None, ind.t()
