# Author: Ketan Fatania
# Affiliation: University of Bath
# Email: kf432@bath.ac.uk
# GitHub: https://github.com/ketanfatania/
# Created: 2023-03-06
# Last Modified: 2023-03-23
#                2025-09-12 - PM - Removing unnecessary code required for MRF-DiPh
from torch import Tensor

import torch
import numpy as np
import scipy.io as scio

from sklearn.neighbors import KDTree


def load_dict(file_path: str) -> Tensor:
    """
    Author: Ketan Fatania ;
    Affiliation: University of Bath ;
    Email: kf432@bath.ac.uk ;
    GitHub: https://github.com/ketanfatania/ ;
    """

    dict_data = scio.loadmat(file_path)
    dict_data = dict_data['dict']
    # print(type(dict_data))                    # <class 'numpy.ndarray'>
    # print(dict_data.dtype)                    # [('lut', 'O'), ('D', 'O'), ('normD', 'O'), ('V', 'O')] - for svd compressed dictionaries
    # print(dict_data.dtype)                    # [('D_nn', 'O'), ('lut', 'O'), ('V', 'O'), ('D', 'O'), ('normD', 'O')] - for uncompressed dictionary
    # print(dict_data.shape)                    # (1, 1)

    dict_data = dict_data[0, 0]                 # To make keys accessible
    # print(type(dict_data))                    # <class 'numpy.void'>
    # print(dict_data.dtype)                    # [('lut', 'O'), ('D', 'O'), ('normD', 'O'), ('V', 'O')] - for svd compressed dictionaries
    # print(dict_data.dtype)                    # [('D_nn', 'O'), ('lut', 'O'), ('V', 'O'), ('D', 'O'), ('normD', 'O')] - for uncompressed dictionary
    # print(dict_data.shape)                    # ()

    return dict_data

def qmaps_to_tsmis_kdtree_svd(qmaps: Tensor,
                              dict_path: str):

    """
    Author: Ketan Fatania ;
    Translated from: Mohammad Golbabaee's Matlab Code ;
    Affiliation: University of Bath ;
    Email: kf432@bath.ac.uk ;
    GitHub: https://github.com/ketanfatania/ ;
    """
    b, k, rows, cols = qmaps.shape
    # ---- Prepare T1 and T2
    t1_t2 = qmaps[0, 0:2, ...]
    # print(type(t1_t2))                          # <class 'torch.Tensor'>
    # print(t1_t2.dtype)                          # torch.float32
    # print(t1_t2.shape)                          # torch.Size([2, 230, 230])
    t1_t2 = torch.reshape(t1_t2, (t1_t2.shape[0], -1))
    # print(type(t1_t2))                          # <class 'torch.Tensor'>
    # print(t1_t2.dtype)                          # torch.float32
    # print(t1_t2.shape)                          # torch.Size([2, 52900])
    t1_t2 = torch.transpose(t1_t2, dim0=0, dim1=1)
    # print(type(t1_t2))                          # <class 'torch.Tensor'>
    # print(t1_t2.dtype)                          # torch.float32
    # print(t1_t2.shape)                          # torch.Size([52900, 2])

    # ---- Load Lookup table
    dict_data = load_dict(dict_path)
    # print(type(dict_data))                    # <class 'numpy.void'>
    # print(dict_data.dtype)                    # [('lut', 'O'), ('D', 'O'), ('normD', 'O'), ('V', 'O')]
    # print(dict_data.shape)                    # ()
    lut = dict_data['lut']
    # print(type(lut))                    # <class 'numpy.ndarray'>
    # print(lut.dtype)                    # float64
    # print(lut.shape)                    # (94777, 2)

    # ---- Match T1 and T2 values with signal evolutions (fingerprints)
    tree = KDTree(lut, leaf_size=40)            # default leaf_size=40
    dist, ind = tree.query(t1_t2, k=1)
    # print(dist.shape)                       # (52900, 1)
    # print(ind.shape)                        # (52900, 1)
    ind = ind[:, 0]
    # print(ind.shape)                        # (52900,)

    # ---- Load SVD dictionary and truncate with num_time_frames
    mag_values = dict_data['D']         # magnetisation values for SVD compressed 10 timeframes from 1000 timeframes
    # print(type(mag_values))                    # <class 'numpy.ndarray'>
    # print(mag_values.dtype)                    # float64
    # print(mag_values.shape)                    # (94777, 10)
    tsmis = mag_values[ind, :]
    # print(type(tsmis))                          # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                          # float64
    # print(tsmis.shape)                          # (52900, 10)

    # ---- UnNormalise TSMIs because dictionary was normalised
    normD = dict_data['normD']
    # print(type(normD))                    # <class 'numpy.ndarray'>
    # print(normD.dtype)                    # float64
    # print(normD.shape)                    # (94777, 1)
    tsmis = tsmis * normD[ind]
    # print(type(tsmis))                              # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                              # float64
    # print(tsmis.shape)                              # (52900, 10)

    # ---- Reshape and set dtype
    tsmis = np.transpose(tsmis)
    # print(type(tsmis))                                  # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                                  # float64
    # print(tsmis.shape)                                  # (10, 52900)
    tsmis = np.reshape(tsmis, (10, rows, cols))
    # print(type(tsmis))                                  # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                                  # float64
    # print(tsmis.shape)                                  # (10, 230, 230)
    tsmis = tsmis.astype('complex64')                     # TODO add option for this
    # print(type(tsmis))                                  # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                                  # complex64
    # print(tsmis.shape)                                  # (10, 230, 230)

    # ---- Introduce phase using pd maps and broadcasting
    pd_complex = qmaps[0, 2, ...] + (1j * qmaps[0, 3, ...])
    # print(type(pd_complex))                             # <class 'torch.Tensor'>
    # print(pd_complex.dtype)                             # torch.complex64
    # print(pd_complex.shape)                             # torch.Size([230, 230])
    tsmis = (torch.from_numpy(tsmis) * pd_complex).unsqueeze(0)
    # print(type(tsmis))                                  # <class 'torch.Tensor'>
    # print(tsmis.dtype)                                  # torch.complex64
    # print(tsmis.shape)                                  # torch.Size([1, 10, 230, 230])

    return tsmis


def qmaps_to_tsmis_kdtree_uncompressed(qmaps: Tensor,
                                       dict_path: str,
                                       num_time_frames: int = 1000):

    """
    Author: Ketan Fatania ;
    Translated from: Mohammad Golbabaee's Matlab Code ;
    Affiliation: University of Bath ;
    Email: kf432@bath.ac.uk ;
    GitHub: https://github.com/ketanfatania/ ;
    """

    if len(qmaps.shape) == 4:
        _, _, height, width = qmaps.shape
    else:
        raise Exception('Expected 4 dimensions')

    # ---- Prepare T1 and T2
    t1_t2 = qmaps[0, 0:2, ...]
    # print(type(t1_t2))                          # <class 'torch.Tensor'>
    # print(t1_t2.dtype)                          # torch.float32
    # print(t1_t2.shape)                          # torch.Size([2, 230, 230])
    t1_t2 = torch.reshape(t1_t2, (t1_t2.shape[0], -1))
    # print(type(t1_t2))                          # <class 'torch.Tensor'>
    # print(t1_t2.dtype)                          # torch.float32
    # print(t1_t2.shape)                          # torch.Size([2, 52900])
    t1_t2 = torch.transpose(t1_t2, dim0=0, dim1=1)
    # print(type(t1_t2))                          # <class 'torch.Tensor'>
    # print(t1_t2.dtype)                          # torch.float32
    # print(t1_t2.shape)                          # torch.Size([52900, 2])

    # ---- Load Lookup table
    dict_data = load_dict(dict_path)
    # print(type(dict_data))                    # <class 'numpy.void'>
    # print(dict_data.dtype)                    # [('D_nn', 'O'), ('lut', 'O'), ('V', 'O'), ('D', 'O'), ('normD', 'O')]
    # print(dict_data.shape)                    # ()
    lut = dict_data['lut']
    # print(type(lut))                    # <class 'numpy.ndarray'>
    # print(lut.dtype)                    # float64
    # print(lut.shape)                    # (94777, 2)

    # ---- Match T1 and T2 values with signal evolutions (fingerprints)
    tree = KDTree(lut, leaf_size=40)            # default leaf_size=40
    dist, ind = tree.query(t1_t2, k=1)
    # print(dist.shape)                       # (52900, 1)
    # print(ind.shape)                        # (52900, 1)
    ind = ind[:, 0]
    # print(ind.shape)                        # (52900,)

    # ---- Load full uncompressed dictionary and truncate with num_time_frames
    mag_values = dict_data['D_nn'][:, 0:num_time_frames]         # magnetisation values for uncompressed timeframes
    # print(type(mag_values))                    # <class 'numpy.ndarray'>
    # print(mag_values.dtype)                    # float64
    # print(mag_values.shape)                    # (94777, 1000) for num_time_frames == 1000 ; (94777, 200) for num_time_frames == 200
    tsmis = mag_values[ind, :]
    # print(type(tsmis))                          # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                          # float64
    # print(tsmis.shape)                          # (52900, 1000) ; (52900, 200)

    # ---- UnNormalise TSMIs because dictionary was normalised
    normD = dict_data['normD']
    # print(type(normD))                    # <class 'numpy.ndarray'>
    # print(normD.dtype)                    # float64
    # print(normD.shape)                    # (94777, 1)
    tsmis = tsmis * normD[ind]
    # print(type(tsmis))                              # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                              # float64
    # print(tsmis.shape)                              # (52900, 1000) ; (52900, 200)

    # ---- Reshape and set dtype
    tsmis = np.transpose(tsmis)
    # print(type(tsmis))                                  # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                                  # float64
    # print(tsmis.shape)                                  # (1000, 52900) ; (200, 52900)
    tsmis = np.reshape(tsmis, (num_time_frames, height, width))
    # print(type(tsmis))                                  # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                                  # float64
    # print(tsmis.shape)                                  # (1000, 230, 230) ; (200, 230, 230)
    tsmis = tsmis.astype('float32')                       # TODO: add option
    # print(type(tsmis))                                  # <class 'numpy.ndarray'>
    # print(tsmis.dtype)                                  # float32
    # print(tsmis.shape)                                  # (1000, 230, 230) ; (200, 230, 230)

    # ---- Introduce phase using pd maps and broadcasting
    pd_complex = qmaps[0, 2, ...] + (1j * qmaps[0, 3, ...])
    # print(type(pd_complex))                             # <class 'torch.Tensor'>
    # print(pd_complex.dtype)                             # torch.complex64
    # print(pd_complex.shape)                             # torch.Size([230, 230])
    tsmis = (torch.from_numpy(tsmis) * pd_complex).unsqueeze(0)
    # print(type(tsmis))                                  # <class 'torch.Tensor'>
    # print(tsmis.dtype)                                  # torch.complex64
    # print(tsmis.shape)                                  # torch.Size([1, 1000, 230, 230]) ; torch.Size([1, 200, 230, 230])

    return tsmis
