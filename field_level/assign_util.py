# !/usr/bin/env python3

import sys
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

import field_level.coord as coord

@partial(jit, static_argnums=(4, 5, 6))
def assign(boxsize, field, weight, pos, window_order, interlace=0, contd=0):
    ng = field.shape[0]
    cell_size = boxsize / ng
    if len(pos.shape) == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)
    num = pos.shape[-1]
    pos_mesh = pos / cell_size

    if interlace:
        pos_mesh += 0.5

    field = assign_(field, pos_mesh, weight, ng, window_order)

    if not contd:
        num_particles = pos.shape[-1] if pos.ndim == 2 else jnp.prod(pos.shape[-3:])
        field /= num_particles / (ng ** 3)

    return field

@partial(jit, static_argnums=(3, 4,))
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

    # periodic boundary conditions; do not use module, it is not compatible with auto-grad
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

    def update_field(i, f):
        indices = i + shifts
        indices = jnp.where(indices < 0, indices + ng, indices)
        indices = jnp.where(indices >= ng, indices - ng, indices)
        weights = vmap(lambda shift: compute_weights(f, shift))(shifts)
        return indices, weights * weight

    indices_weights = vmap(update_field)(imesh.T, fmesh.T)

    indices = indices_weights[0].reshape(-1, 3)
    weights = indices_weights[1].reshape(-1)

    field = field.at[indices[:, 0], indices[:, 1], indices[:, 2]].add(weights)
    return field
