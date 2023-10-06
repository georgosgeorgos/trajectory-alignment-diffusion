import numpy as np
import torch as th

def compute_kernel_load(batch_load_sample, axis):
    size = batch_load_sample.size(-1)
    if axis == "x":
        ix = 0
        xx = th.argwhere(batch_load_sample[0] != 0)
        coord = xx
    elif axis == "y":
        ix = 1
        yy = th.argwhere(batch_load_sample[1] != 0)
        coord = yy

    if len(coord) == 0:
        return batch_load_sample[ix], []

    x_grid = th.tensor([i for i in range(size)])
    y_grid = th.tensor([j for j in range(size)])

    kernel_load = 0
    for l in range(len(coord)):
        x_grid = th.tensor([i for i in range(size)])
        y_grid = th.tensor([j for j in range(size)])
        # distance
        x_grid = x_grid - coord[l][0]
        y_grid = y_grid - coord[l][1]

        grid = th.meshgrid(x_grid, y_grid)

        r_load = th.sqrt(grid[0]**2 + grid[1]**2)

        if axis == "x":
            p = batch_load_sample[0][coord[l][0], coord[l][1]]
        elif axis == "y":
            p = batch_load_sample[1][coord[l][0], coord[l][1]]

        kernel = 1 - th.exp(- 1/r_load**2)
        kernel_load += kernel * p
        
    #kernel_load = kernel_load / kernel_load.max() 
    return kernel_load, coord

def compute_kernel_boundary(batch_boundary_sample, axis):
    size = batch_boundary_sample.size(-1)
    if axis == "x":
        ix = 0
        xx = th.argwhere(batch_boundary_sample[0] != 0)
        coord = xx
    elif axis == "y":
        ix = 1 
        yy = th.argwhere(batch_boundary_sample[1] != 0)
        coord = yy

    if len(coord) == 0:
        return batch_boundary_sample[ix], []    

    x_grid = th.tensor([i for i in range(size)])
    y_grid = th.tensor([j for j in range(size)])

    kernel_boundary = 0
    for l in range(len(coord)):
        x_grid = th.tensor([i for i in range(size)])
        y_grid = th.tensor([j for j in range(size)])
        # distance
        x_grid = x_grid - coord[l][0]
        y_grid = y_grid - coord[l][1]

        grid = th.meshgrid(x_grid, y_grid)

        r_boundary = th.sqrt(grid[0]**2 + grid[1]**2)
        
        if axis == "x":
            bc = batch_boundary_sample[0][coord[l][0], coord[l][1]]
        elif axis == "y":
            bc = batch_boundary_sample[1][coord[l][0], coord[l][1]]
        
        kernel = 1 - th.exp(- 1/r_boundary**2)
        kernel_boundary += kernel * bc
        
    kernel_boundary = kernel_boundary / kernel_boundary.max()    
    return kernel_boundary, coord



