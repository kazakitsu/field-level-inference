# !/usr/bin/env python3

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

import field_level.coord as coord

@partial(jit, static_argnames=('window_order', 'interlace', 'contd', 'max_scatter_indices'))
def assign(boxsize, field, weight, pos, 
           window_order, 
           interlace=0, 
           contd=0, 
           max_scatter_indices=500_000_000):
    '''
    boxsize : float
    field   : 3D array, shape = (ng, ng, ng)
    weight  : 1D array, shape = (num_particles,) or (Nx, Ny, Nz)
    pos     : 3D array, shape = (3, num_particles) or (3, Nx, Ny, Nz)
    window_order : int, 1(ngp), 2(cic), or 3(tsc)
    interlace : bool, 0 or 1
    contd : bool, 0 or 1
    max_scatter_indices : int, the max number of scatter indices; if the number of scatter indices exceeds this value, use chunked scatter
    '''
    ng = field.shape[0]
    cell_size = boxsize / ng

    if len(pos.shape) == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    num = pos.shape[-1]
    pos_mesh = pos / cell_size

    if interlace:
        pos_mesh += 0.5

    if window_order == 1:
        n_shifts = 1
    elif window_order == 2:
        n_shifts = 8
    elif window_order == 3:
        n_shifts = 27
    else:
        raise ValueError(f"Unsupported window_order={window_order}")

    total_scatter = num * n_shifts

    def single_assign(field, pos_mesh, weight):
        return assign_(field, pos_mesh, weight, ng, window_order)

    #field = assign_(field, pos_mesh, weight, ng, window_order)

    if total_scatter > max_scatter_indices:
        chunk_size = max_scatter_indices // n_shifts
        if chunk_size < 1:
            chunk_size = 1

        start = 0
        while start < num:
            end = min(start + chunk_size, num)

            pos_chunk = pos_mesh[..., start:end]
            w_chunck  = weight[start:end]

            field = single_assign(field, pos_chunk, w_chunck)
            start = end

    else:
        field = single_assign(field, pos_mesh, weight)

    if not contd:
        num_particles = pos.shape[-1] if pos.ndim == 2 else jnp.prod(pos.shape[-3:])
        field /= num_particles / (ng ** 3)

    return field

@partial(jit, static_argnames=('ng', 'window_order',))
def assign_(field, pos_mesh, weight, ng, window_order):
    if window_order == 1:  # NGP
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = jnp.zeros_like(pos_mesh)
        shifts = jnp.array([[0, 0, 0]])
    elif window_order == 2:  # CIC
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = pos_mesh - imesh
        shifts = jnp.stack(jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2), indexing='ij'), -1).reshape(-1, 3)
    elif window_order == 3:  # TSC
        imesh = jnp.floor(pos_mesh - 1.5).astype(jnp.int32) + 2
        fmesh = pos_mesh - imesh
        shifts = jnp.stack(jnp.meshgrid(jnp.arange(-1, 2), jnp.arange(-1, 2), jnp.arange(-1, 2), indexing='ij'), -1).reshape(-1, 3)
    else:
        raise ValueError(f"Unsupported window_order={window_order}")

    # periodic boundary conditions; do not use modulo, it is not compatible with auto-grad
    imesh = jnp.where(imesh < 0, imesh + ng, imesh)
    imesh = jnp.where(imesh >= ng, imesh - ng, imesh)

    def compute_weights(fmesh, shift):
        if window_order == 1:  # NGP 
            return 1.0
        elif window_order == 2:  # CIC 
            w = jnp.where(shift == 0, 1.0 - fmesh, fmesh)
        elif window_order == 3:  # TSC 
            w = jnp.where(
                shift == 0,
                0.75 - fmesh**2,
                0.5 * (fmesh**2 + shift * fmesh + 0.25)
            )
        return jnp.prod(w, axis=-1)

    def update_field(i, f, w):
        indices = i + shifts
        indices = jnp.where(indices < 0, indices + ng, indices)
        indices = jnp.where(indices >= ng, indices - ng, indices)
        # compute window weights
        w_shifts = vmap(lambda shift: compute_weights(f, shift))(shifts)
        return indices, w_shifts * w

    indices_weights = vmap(update_field, in_axes=(0,0,0))(imesh.T, fmesh.T, weight)

    indices = indices_weights[0].reshape(-1, 3)
    weights = indices_weights[1].reshape(-1)

    field = field.at[indices[:, 0], indices[:, 1], indices[:, 2]].add(weights)
    return field
