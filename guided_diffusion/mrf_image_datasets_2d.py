import os
import torch
import numpy as np
import torchvision.transforms.functional as F

from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset


cond_factors = {3: 0.000004}
ref_factors = {3: 0.004}

def load_data(
    *, image_size, input_dir, ref_dir, batch_size, vols, slices, deterministic=False, augment=True,
     cut=3, use_condition=True):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not input_dir:
        raise ValueError("unspecified data directory")
    all_files = []
    for v in vols:
        for s in slices:
            all_files.append(f'vol{v}_slice{s}')

    dataset = MRFDataset(
        image_size=image_size,
        input_dir=input_dir,
        ref_dir=ref_dir,
        image_names=all_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        augment=augment,
        fisp_cut=cut,
        use_condition=use_condition
    )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=not deterministic, num_workers=1, drop_last=True
    )
    while True:
        yield from loader


class MRFDataset(Dataset):
    def __init__(self, image_size, ref_dir, input_dir, image_names, fisp_cut, shard=0, num_shards=1,
                  augment=True, use_condition=True):
        super().__init__()
        # paths are:
        # TSMI Adjoint: ./datasets/Deli-CS/adjoint_tsmi/ (filename: adjoint_tsmi_volX_sliceY_cutC.npz)
        # TSMI Reference: ./datasets/Deli-CS/reference_tsmi/ (filename: tsmi_ref_volX_sliceY.npz)
        self.input_dir = input_dir
        self.ref_dir = ref_dir
        self.local_images = image_names[shard:][::num_shards]
        self.image_size = image_size
        self.augment = augment
        self.fisp_cut = fisp_cut
        self.ref_factor = ref_factors[fisp_cut]
        self.cond_factor = cond_factors[fisp_cut]
        self.use_condition = use_condition

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        x_c_path = os.path.join(self.input_dir, f'adjoint_tsmi_{self.local_images[idx]}_cut{self.fisp_cut}.npz')
        x_c = np.load(x_c_path)['tsmi_adjoint']
        x_c = torch.from_numpy(x_c).squeeze()
        ksvd, h, w = x_c.shape

        x_c = torch.view_as_real(x_c).permute(0, 3, 1, 2).contiguous()
        x_c = x_c.view(ksvd*2, h, w)
        ref_path = os.path.join(self.ref_dir, f'tsmi_ref_{self.local_images[idx]}_cut{self.fisp_cut}.npz')
        ref = np.load(ref_path)['tsmi_ref']
        ref = torch.from_numpy(ref).squeeze()

        ref = torch.view_as_real(ref).permute(0, 3, 1, 2).contiguous()
        ref = ref.view(ksvd * 2, h, w)

        # Image normalization in (roughly) range [-1, 1]
        x_c = x_c / self.cond_factor
        ref = ref / self.ref_factor

        if h < self.image_size:
            padding = (self.image_size - h)//2
            ref = torch.nn.functional.pad(ref, [padding] * 4)
            x_c = torch.nn.functional.pad(x_c, [padding] * 4)
        elif h > self.image_size:
            top_left = (h // 2) - (self.image_size // 2)
            top_left = top_left, top_left
            ref = ref[:, top_left[0]:top_left[0] + self.image_size, top_left[1]:top_left[1] + self.image_size]
            x_c = x_c[:, top_left[0]:top_left[0] + self.image_size, top_left[1]:top_left[1] + self.image_size]

        if self.augment:
            if np.random.rand() < 0.5:  # Horizontal Flip
                ref = F.hflip(ref)
                x_c = F.hflip(x_c)
            if np.random.rand() < 0.5:  # Vertical Flip
                ref = F.vflip(ref)
                x_c = F.vflip(x_c)

        out_dict = {}
        if self.use_condition:
            out_dict['low_res'] = x_c
        return ref, out_dict

