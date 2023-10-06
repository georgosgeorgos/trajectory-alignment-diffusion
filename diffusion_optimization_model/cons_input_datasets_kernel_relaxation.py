import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from diffusion_optimization_model.kernel_relaxation import compute_kernel_load, compute_kernel_boundary
import torch as th


def load_data(
    *, data_dir, deterministic=True, compliance_conditioning=True
):
    """
    Create the generator used for sampling.
    The dataset should contain:
    - the .npy files of physical fields in the form cons_pf_array_X.npy,
    - the .npy files of loads in the form cons_load_array_X.npy,
    - the .npy files of boundary conditions in the form cons_bc_array_X.npy.

    :param data_dir: the dataset directory.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_input_constraints, all_input_raw_loads, all_input_raw_BCs = _list_input_files_recursively(data_dir)
    dataset = InputConstraintsDataset(
        all_input_constraints,
        all_input_raw_loads,
        all_input_raw_BCs,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        compliance_conditioning=compliance_conditioning,
    )
    if deterministic:
        loader = DataLoader(
            dataset, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_input_files_recursively(data_dir):
    input_constraints = []
    input_raw_loads = []
    input_raw_BCs = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            if "load" in entry:
                input_raw_loads.append(full_path) # Load file
            elif "bc" in entry:
                input_raw_BCs.append(full_path) # BC file
            else:
                input_constraints.append(full_path) # Physical fields file
        elif bf.isdir(full_path):
            input_constraints.extend(_list_input_files_recursively(full_path))
            input_raw_loads.extend(_list_input_files_recursively(full_path))
            input_raw_BCs.extend(_list_input_files_recursively(full_path))
    return input_constraints, input_raw_loads, input_raw_BCs


class InputConstraintsDataset(Dataset):
    def __init__(self, input_constraints_paths, input_raw_loads_paths, input_raw_BCs_paths, shard=0, num_shards=1,
                compliance_conditioning=True):
        super().__init__()
        self.local_input_constraints = input_constraints_paths[shard:][::num_shards]
        self.local_input_raw_loads = input_raw_loads_paths[shard:][::num_shards]
        self.local_input_raw_BCs = input_raw_BCs_paths[shard:][::num_shards]
        self.compliance_conditioning = compliance_conditioning

    def __len__(self):
        return len(self.local_input_constraints)

    def __getitem__(self, idx, mode="kernel+"):
        input_constraints_path = self.local_input_constraints[idx]
        input_raw_loads_path = self.local_input_raw_loads[idx]
        input_raw_BCs_path = self.local_input_raw_BCs[idx]
      
        input_constraints = np.load(input_constraints_path)
        input_raw_loads = np.load(input_raw_loads_path)
        input_raw_BCs = np.load(input_raw_BCs_path)
        
        input_constraints = np.transpose(input_constraints, [2, 0, 1]).astype(np.float32)
        input_raw_loads = np.transpose(input_raw_loads, [2, 0, 1]).astype(np.float32)
        input_raw_BCs = np.transpose(input_raw_BCs, [2, 0, 1]).astype(np.float32)
        
        
        vf = input_constraints[0, :, :]
        vf = th.from_numpy(vf)
        
        loads = th.from_numpy(input_raw_loads)
        if mode in ["kernel", "kernel+"]:
            kernel_load_xx, _ = compute_kernel_load(loads, axis="x")
            kernel_load_yy, _ = compute_kernel_load(loads, axis="y")
            kernel_load = th.cat([kernel_load_xx.unsqueeze(0), kernel_load_yy.unsqueeze(0)], 0)
        else:
            kernel_load = loads
        
        bcs = th.from_numpy(input_raw_BCs)
        if mode in ["kernel", "kernel+"]:
            kernel_boundary_xx, _ = compute_kernel_boundary(bcs, axis="x")
            kernel_boundary_yy, _ = compute_kernel_boundary(bcs, axis="y")
            kernel_boundary = th.cat([kernel_boundary_xx.unsqueeze(0), kernel_boundary_yy.unsqueeze(0)], 0)
        else:
            kernel_boundary = bcs
        
        input_constraints = vf.unsqueeze(0)
#         if self.compliance_conditioning:
#             # small value
#             opt_compliance = 0.01
#             compliance = th.ones_like(vf) * opt_compliance
#             input_constraints = th.cat([compliance.unsqueeze(0), input_constraints], axis = 0)
            
        if mode == "kernel":
            _constraints = th.cat([kernel_load, kernel_boundary], 0)
        elif mode == "kernel+":
            _constraints = th.cat([loads, bcs, kernel_load, kernel_boundary], 0)
            
        return input_constraints, _constraints, bcs
    
    
    
    
    