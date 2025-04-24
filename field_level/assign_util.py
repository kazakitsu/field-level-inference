#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial

@partial(jit, static_argnames=('num_particles', 'window_order', 'interlace', 'contd', 'max_scatter_indices'))
def assign(boxsize, field, weight, pos, num_particles, window_order, interlace=0, contd=0, max_scatter_indices=100_000_000):
    """
    Parameters:
      boxsize : float
      field   : 3D array, shape = (ng, ng, ng)
      weight  : 1D array, shape = (num_particles,) or (Nx, Ny, Nz)
      pos     : 3D array, shape = (3, num_particles) or (3, Nx, Ny, Nz)
      window_order : int, 1 (ngp), 2 (cic), or 3 (tsc)
      interlace : bool, 0 or 1
      contd : bool, 0 or 1
      max_scatter_indices : int, if number of scatter indices exceeds this, chunked scatter is used.
    """
    ng = field.shape[0]
    cell_size = boxsize / ng

    if len(pos.shape) == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)

    #num = pos.shape[-1]
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

    total_scatter = num_particles * n_shifts

    def single_assign(field, pos_mesh, weight):
        return assign_(field, pos_mesh, weight, ng, window_order)

    # If the total number of scatter indices exceeds the maximum,
    # use a static fori_loop to process the inputs in chunks.
    if total_scatter > max_scatter_indices:
        # static chunk size
        chunk_size = max(max_scatter_indices // n_shifts, 1)
        num_iters = (num_particles + chunk_size - 1) // chunk_size

        # Pad pos_mesh and weight so that dynamic_slice is always in-bound
        pos_padded = jnp.pad(pos_mesh, ((0, 0), (0, chunk_size)), constant_values=0.0)
        weight_padded = jnp.pad(weight, (0, chunk_size), constant_values=0.0)

        # Loop over chunks, each of fixed length chunk_size
        def chunk_body(i, f):
            start = i * chunk_size
            # dynamic_slice with static slice size
            pos_ck = lax.dynamic_slice_in_dim(pos_padded, start, chunk_size, axis=-1)
            w_ck   = lax.dynamic_slice_in_dim(weight_padded, start, chunk_size, axis=0)
            return single_assign(f, pos_ck, w_ck)

        field = lax.fori_loop(0, num_iters, chunk_body, field)
    else:
        field = single_assign(field, pos_mesh, weight)

    if not contd:
        #num_particles = pos.shape[-1] if pos.ndim == 2 else jnp.prod(pos.shape[-3:])
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

    # Periodic boundary conditions
    imesh = jnp.where(imesh < 0, imesh + ng, imesh)
    imesh = jnp.where(imesh >= ng, imesh - ng, imesh)

    def compute_weights(fmesh, shift):
        if window_order == 1:    # NGP
            return 1.0
        elif window_order == 2:  # CIC
            w = jnp.where(shift == 0, 1.0 - fmesh, fmesh)
        elif window_order == 3:  # TSC
            w = jnp.where(shift == 0, 0.75 - fmesh**2, 0.5 * (fmesh**2 + shift * fmesh + 0.25))
        return jnp.prod(w, axis=-1)

    def update_field(i, f, w):
        indices = i + shifts
        indices = jnp.where(indices < 0, indices + ng, indices)
        indices = jnp.where(indices >= ng, indices - ng, indices)
        w_shifts = vmap(lambda shift: compute_weights(f, shift))(shifts)
        return indices, w_shifts * w

    indices_weights = vmap(update_field, in_axes=(0, 0, 0))(imesh.T, fmesh.T, weight)
    indices = indices_weights[0].reshape(-1, 3)
    weights = indices_weights[1].reshape(-1)

    field = field.at[indices[:, 0], indices[:, 1], indices[:, 2]].add(weights)
    return field
