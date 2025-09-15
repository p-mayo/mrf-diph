import scipy.io as sio
import numpy as np
import torch
import torchkbnufft as tkbn

def subsamp(x, R, mode):
    if mode == 'regular':
        return x[..., ::R]
    elif mode =='roll':
        sub = torch.zeros_like(x[..., ::R])
        for i in range(R):
            sub[..., i::R, :] =x[..., i::R, i::R]
        return sub
    else:
        raise ValueError('invalid subsamp mode')

def reshape_trajectories(traj, time_frames, num_samples,  norm=1.):
    if traj.shape[0]< num_samples *time_frames:
        traj = traj.reshape((num_samples,-1,2), order='F')
        n_int = traj.shape[1]
        NEX = np.ceil(time_frames/n_int)
        traj = np.tile(traj,(1,int(NEX),1))
        traj = traj[:,:time_frames].reshape((-1,2), order='F')
    return traj[:num_samples *time_frames,:]

def load_traj_basis_dcf_2dscans(acquisition, cut=None):
    if acquisition == 'fisp':
        num_samples = 920
        num_interleaves = 377  # 377
        im_shape = np.array([230., 230.], dtype=int)
        if cut==0:
            time_frames = 1000
        elif cut==1:
            time_frames = 500
        elif cut == 2:
            time_frames = 300
        elif cut==3:
            time_frames = 200
        elif cut==4:
            time_frames = 100

        # kspace_dim = min(num_samples, num_interleaves) * time_frames
        spiral_traj_dcf_path = './mrf_processing/traj_dcf.mat'
        dict_path = f'./mrf_processing/SVD_dict_FISP_cut{cut}.mat'

    trajectory = sio.loadmat(spiral_traj_dcf_path, simplify_cells=True)
    ktraj = trajectory['traj'].astype(np.float32) * 2 * np.pi
    ktraj = reshape_trajectories(ktraj, time_frames, num_samples)
    ktraj = (ktraj).T

    ## load kbnufft's pre-computed dcf
    tmp = trajectory['dcf'].astype(np.float32)
    dcf=[]
    for i in range(time_frames):
        dcf = np.concatenate((dcf,tmp),axis=0)
    #np.tile(dcf, (1,time_frames)).reshape((-1,1),order='F').T
    dcf = dcf.transpose()


    # ---load MRF's SVD basis --------
    dict = sio.loadmat(dict_path)['dict']
    V = dict['V'][0,0]
    basis = V.transpose()

    return ktraj, dcf, basis, time_frames, num_samples, im_shape, dict_path

class NUFFT_Pytorch:
    def __init__(self, im_shape, trajectory, basis, dcf=None, device='cpu', lib_nufft ='tkbn'):
        self.im_shape = np.array(im_shape)
        self.is3D = (im_shape.ndim == 3)
        self.ktraj = trajectory.to(device)  # Values should be within -pi/2 to pi/2
        self.basis = basis.to(device)
        self.k, self.NT = self.basis.shape # NT is mrf's number_timeframes
        self.dsamp = int(self.ktraj.shape[1] / self.NT) # dsamp is mrf's number of kspace samples at each timeframe
        if dcf is not None: #dcf size should be [NT dsamp]
            # self.dcf = dcf.to(torch.float32).to(device).reshape(self.dsamp,self.NT).permute(1,0)
            dcf = dcf.numpy().reshape((self.dsamp, self.NT), order = 'F')
            self.sqrt_dcf = torch.sqrt(torch.tensor(dcf, dtype = torch.float32, device = device)).permute(1,0)
            self.dcf = torch.tensor(dcf, dtype=torch.float32, device=device).permute(1, 0)
        else:
            self.dcf = 1.
            self.sqrt_dcf = 1.

        if lib_nufft == 'tkbn':
            #==== initialise tkbn nufft as backbone
            oversamp = 2
            width = 2.34
            numpoints = 6
            grid_size = (self.im_shape * oversamp).astype(int)
            F = tkbn.KbNufft(im_size=self.im_shape, grid_size=grid_size, numpoints=numpoints, kbwidth=width).to(device)
            FH = tkbn.KbNufftAdjoint(im_size=self.im_shape, grid_size=grid_size, numpoints=numpoints,
                                          kbwidth=width).to(device)
            self.F = lambda x: F(x, self.ktraj, norm='ortho', smaps=None)
            self.FH = lambda y: FH(y, self.ktraj, norm='ortho')

        else:
            # init torch finufft as backbone
            import pytorch_finufft
            # upsampfac = 1.125
            upsampfac = 2.0
            if upsampfac == 2.0:
                kwargs = {}
            else:
                kwargs = {'upsampfac': upsampfac, 'gpu_kerevalmeth': 0, 'gpu_method': 1}

            if device != 'cpu':
                gpuid = int(device.split(':')[1])
                print(gpuid)

                self.F = lambda x: pytorch_finufft.functional.finufft_type2(self.ktraj, x,
                                                         modeord=0, isign=-1, gpu_device_id=gpuid,
                                                         **kwargs) / (2 * self.im_shape[0])
                self.FH = lambda y: pytorch_finufft.functional.finufft_type1(self.ktraj, y, tuple(self.im_shape.tolist()),
                                                         isign=1, modeord=0, gpu_device_id=gpuid,
                                                         **kwargs) / (2 * self.im_shape[0])
                # self.FH = lambda y: pytorch_finufft.functional.finufft_type1(self.ktraj, y,
                #                                                              tuple(self.im_shape.tolist())) / (2 * self.im_shape[0])
        #============

    def fwd(self, x, sens, loop_over_coils = False):
        '''''
        Computes MRF forward operator (not including dcf). Inputs are:
        x.shape = [n_batch, ch_svd, H, W] or [n_batch, ch_svd, H, W, D] torch tensor
        sens.shape = [n_batch, ncoils, H, W] or [n_batch, ncoils, H, W, D] torch tensor
        '''''
        batch_size = x.shape[0]
        if sens is not None:
            n_coils = sens.shape[1]
        else:
            n_coils = 1

        if self.is3D:
            x = x.permute(1, 0, 2, 3, 4).unsqueeze(2)  # size [ch_svd nbatch 1 H W D]
        else:
            x = x.permute(1, 0, 2, 3).unsqueeze(2)  # size [ch_svd nbatch 1 H W]

        if sens is not None:
            x = x * sens  # size [ch_svd nbatch ncoil H W D]

        x = x.reshape( (batch_size * self.k, n_coils)+ tuple(self.im_shape) )

        if loop_over_coils == True:
            y = torch.zeros(size=(self.k * batch_size, n_coils, self.NT * self.dsamp), dtype=torch.complex64,
                            device=x.device)
            for c in range(n_coils):
                y[:, c] = self.F(x[:, c].unsqueeze(1)).squeeze(1)
            x = y
        else:
            x = self.F(x)

        x = x.reshape((self.k, batch_size * n_coils, self.NT, self.dsamp))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((self.k, self.NT, batch_size, n_coils, self.dsamp))
        x = (self.basis[:, :, None, None, None].conj() * x).sum(axis=0)
        #return x.permute(1, 2, 0, 3)*self.sqrt_dcf  # output:[nbatch ncoils NT dsamp]
        return x.permute(1, 2, 0, 3) # output:[nbatch ncoils NT dsamp]

    def adj(self, y, sens, loop_over_coils = False):
        '''''
        Computes MRF adjoint operator including dcf (if dcf provided). Inputs are:
        y.shape  = [nbatch ncoils NT dsamp] torch tensor
        sens.shape = [n_batch, ncoils, H, W] or [n_batch, ncoils, H, W, D] torch tensor
        '''''
        # y = y*self.sqrt_dcf
        y = y * self.dcf
        batch_size, n_coils = y.shape[:2]

        if self.is3D:  # 3D imaging
            H, W, D = self.im_shape
        else:  # 2D imaging
            H, W = self.im_shape
            D = 1

        y = y.permute(2, 0, 1, 3)  # [NT nbatch ncoils dsamp] [880, 1, 8, 49056] torch.complex64

        y = self.basis[:, :, None, None, None] * y  # [5, 880, 1, 8, 49056] torch.complex64
        y = y.permute(0, 2, 3, 1, 4)  # [5, 1, 8, 880, 49056] torch.complex64
        y = y.reshape((self.k * batch_size, n_coils, -1))  # [5, 8, 43M] torch.complex64

        if sens is not None:
            if loop_over_coils == True:
                xhat = torch.zeros(size=(self.k * batch_size, H, W, D), dtype=torch.complex64, device=y.device)
                for c in range(n_coils):
                    xhat += (self.FH(y[:, c].unsqueeze(1)).reshape((self.k, batch_size)+ tuple(self.im_shape))
                             * (sens[:, c].conj()))
                y = xhat
                del xhat
            else:
                y = self.FH(y)
                y = y.reshape((self.k, batch_size, n_coils) + tuple(self.im_shape))
                y = torch.sum(y * sens.conj(), dim=2)

        else:
            y = self.FH(y)
            y = y.reshape((self.k, batch_size, n_coils) + tuple(self.im_shape)).squeeze(2)

        if self.is3D:
            return y.permute(1, 0, 2, 3, 4)  # output shape: [nbatch, ch_svd, H, W, D]
        else:
            return y.permute(1, 0, 2, 3)  # output shape: [nbatch, ch_svd, H, W]


    def proxf(self, x, AHy, sens, opt, loop_over_coils = False):
        ''''
        iteratively solves:
            proxf(x, rho) := amin_z |y-Az|**2 + rho*|z-x|**2

        opt['rho']: trade-off between data fidelity or finding a solution close to x
        opt['cg_maxiter']: max number of conjugate gradient iters to solve above
        opt['cg_tol']: tolerece of conjugate gradient solver
        opt['x0']: initial point, if None starts from x0=AHy.
        AHy: adjoint_A applied on y (in data fidelity)
        sens: sens maps within A operator.

        M. Golbabaee (m.golbabaee@bristol.ac.uk 2024)
        '''
        if 'x0' not in opt:
            opt['x0'] = AHy.clone()
        return mybasic_cg(mv=lambda u: opt['rho'] * u + self.adj(self.fwd(u, sens, loop_over_coils), sens, loop_over_coils),
                          r=AHy + opt['rho'] * x,
                          x0=opt['x0'].clone(),  # initial solution
                          maxiter=opt['cg_maxiter'], tol=opt['cg_tol'], verbose=False)

def mybasic_cg(mv, r, x0, maxiter, tol, verbose=False):
    '''''
    pytorch solver for the linear system 'M*x=r' using conjugate gradient.
    M is a PSD matrix implicitely defined through 'mv' which is M's forward operator i.e. mv(x) = M*x matrix-vector product.
    x0 is the initial solution.
    M Golbabaee 9/1/2024
    '''''
    with torch.no_grad():
        torch.cuda.empty_cache()

        bnorm = torch.linalg.norm(r)
        rtol = bnorm * tol

        r.sub_(mv(x0))
        p = r.clone()
        rnorm_old = torch.linalg.norm(r) ** 2

        for k in range(maxiter):
            Ap = mv(p)
            alpha = rnorm_old / (p.conj() * Ap).sum()
            x0.add_(alpha * p)  # x0=x0+alpha*p
            r.sub_(alpha * Ap)  # r=r-alpha*Ap
            del Ap
            torch.cuda.empty_cache()
            rnorm = torch.linalg.norm(r) ** 2
            if verbose:
                print(f'cg-iter {k}: abs_residual {rnorm:.4e}, rel_residual {torch.sqrt(rnorm) / bnorm:.4e}')
            if torch.sqrt(rnorm) < rtol:
                break
            beta = rnorm / rnorm_old
            p.add_(r, alpha=1 / beta).mul_(beta)  # p=r+beta*p
            rnorm_old = rnorm

        print(f'cg terminates: iter {k + 1}, rel_residual {torch.sqrt(rnorm) / bnorm:.3e}.')
        torch.cuda.empty_cache()
        return x0








