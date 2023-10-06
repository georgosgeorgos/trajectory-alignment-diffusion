import math
import random

from PIL import Image
import blobfile as bf
import os
from mpi4py import MPI
import numpy as np
import h5py
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
import torch as th
from scipy import sparse
from diffusion_optimization_model.kernel_relaxation import compute_kernel_load, compute_kernel_boundary

def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    deterministic=False,
    compliance_conditioning=True,
):
    """
    Create the generator used for training the regressor predicting the compliance.
    The dataset should contain:
    - the .npy files of physical fields in the form cons_pf_array_X.npy,
    - the .npy files of loads in the form cons_load_array_X.npy,
    - the .npy files of boundary conditions in the form cons_bc_array_X.npy,
    - the .png images of the topologies in the form gt_topo_X.png.

    :param data_dir: the dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size of the images.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_names = _list_image_files(data_dir)
    
    dataset = ImageDataset(
        data_dir,
        image_size,
        all_names,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def _list_image_files(data_dir):
    names = []
    size_dataset = 0
    for file in os.listdir(data_dir + "dataset64_intermediate/"):
        file = file.split(".hdf5")[0]
        file = file.split("_")[-2:]
        file = "_".join(file)
        names.append(file)
        size_dataset +=1
    size_dataset *= 120
    names = sorted(names)
    return names


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        resolution,
        image_paths,
        shard=0,
        num_shards=1,
        compliance_conditioning = True
    ):
        super().__init__()

        if not data_dir:
            raise ValueError("unspecified data directory")
        self.data_dir = data_dir
        self.resolution = resolution
        self.all_names = _list_image_files(data_dir)
        self.compliance_conditioning = compliance_conditioning

    def __len__(self):
        return len(self.all_names) * 120

    def __getitem__(self, idx, mode="kernel+"):
        bucket_idx = idx // 120
        idx = idx % 120

        with h5py.File(self.data_dir + "dataset64_intermediate/" + "rand_diff_sams_64_120_" + self.all_names[bucket_idx] + ".hdf5" , "r") as f:
            assert f["sams"].shape[0] == 120
            img = f["sams"][idx]
            img_intermediate = f["intermediate"][idx]
            performance_intermediate = f["objective"][idx]
        
        # bc_x, bc_y, load_x, load_y
        with open(self.data_dir + "dataset64_intermediate_constraints_sparse/" + "cons_array_64_120_" + self.all_names[bucket_idx] + ".pkl",'rb') as f:
            constraints = pkl.load(f)[idx]
        bcs_x = sparse.csr_matrix.toarray(constraints[0])
        bcs_y = sparse.csr_matrix.toarray(constraints[1])
        bcs_x = np.expand_dims(bcs_x,0)
        bcs_y = np.expand_dims(bcs_y,0)
        bcs = np.concatenate([bcs_x, bcs_y], axis=0)

        loads_x = sparse.csr_matrix.toarray(constraints[2])
        loads_y = sparse.csr_matrix.toarray(constraints[3])
        loads_x = np.expand_dims(loads_x, 0)
        loads_y = np.expand_dims(loads_y, 0)
        loads = np.concatenate([loads_x, loads_y], axis=0)
        
        vf = np.load(self.data_dir + "dataset64_intermediate_configs_summaries/" + "rand_diff_configs_64_120_" + self.all_names[bucket_idx] + ".npy", allow_pickle=True,  encoding="latin1")[idx]
        vf = vf["VOL_FRAC"]
        vf = np.ones(img.shape) * vf
        vf = np.expand_dims(vf, 0)
        
        img = img.astype(np.float32) * 2 - 1
        img = img.reshape(1, self.resolution, self.resolution)

        assert vf.shape[1:2] == img.shape[1:2], "The constraints do not fit the dimension of the image"

        assert loads.shape[1:2] == img.shape[1:2], "The constraints do not fit the dimension of the image"

        assert bcs.shape[1:2] == img.shape[1:2], "The constraints do not fit the dimension of the image"

        out_dict = {}
        
        img_intermediate = img_intermediate.astype(np.float32) * 2 - 1
        loads = loads.astype(np.float32)
        bcs = bcs.astype(np.float32)
        vf = vf.astype(np.float32)

        performance_intermediate = performance_intermediate.astype(np.float32)
        performance_intermediate = th.from_numpy(performance_intermediate)
        
        vf = th.from_numpy(vf)
        
        loads = th.from_numpy(loads)
        if mode in ["kernel", "kernel+"]:
            kernel_load_xx, _ = compute_kernel_load(loads, axis="x")
            kernel_load_yy, _ = compute_kernel_load(loads, axis="y")
            kernel_load = th.cat([kernel_load_xx.unsqueeze(0), kernel_load_yy.unsqueeze(0)], 0)
        else:
            kernel_load = loads
        
        bcs = th.from_numpy(bcs)
        if mode in ["kernel", "kernel+"]:
            kernel_boundary_xx, _ = compute_kernel_boundary(bcs, axis="x")
            kernel_boundary_yy, _ = compute_kernel_boundary(bcs, axis="y")
            kernel_boundary = th.cat([kernel_boundary_xx.unsqueeze(0), kernel_boundary_yy.unsqueeze(0)], 0)
        else:
            kernel_boundary = bcs
        
        if mode == "kernel":
            # 1 + 2 + 2
            constraints = th.cat([vf, kernel_load, kernel_boundary], axis = 0)
        elif mode == "kernel+":
            # 1 + (2 + 2) + (2 + 2)
            constraints = th.cat([vf, loads, bcs, kernel_load, kernel_boundary], axis = 0)
        else:
            # 1 + (2 + 2)
            constraints = th.cat([vf, loads, bcs], axis = 0)

        out_dict = {"objective": performance_intermediate}
        #out_dict["d"] = np.array(self.local_deflections[num_im], dtype=np.float32)
#         if self.compliance_conditioning:
#             compliance = th.ones_like(vf) * out_dict["d"]
#             constraints = th.cat([compliance.unsqueeze(0), constraints], axis = 0)
        return img, constraints, img_intermediate, out_dict

def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

# if __name__ == "__main__":
#     data_dir = "./dataset_dom/"
#     batch_size=32
#     image_size = 64
#     print(data_dir)
#     names = _list_image_files(data_dir)
#     print(len(names))
#     data= load_data(data_dir, batch_size, image_size)
#     data.__next__()
