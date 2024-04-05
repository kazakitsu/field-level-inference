# !/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
import sys
import jax

jax.config.update("jax_enable_x64", True)

import field_level.coord as coord

@partial(jit, static_argnums=(1, 5, 6, ))
def assign(boxsize, ng, weight, pos, field, window_order, interlace=0, contd=0):
    cell_size = boxsize/ng
    if len(pos.shape) == 2:
        num = pos.shape[-1]
    elif len(pos.shape) == 4:
        ng_p = pos.shape[-1]
        num = ng_p*ng_p*ng_p
    ng3 = ng*ng*ng
    pos_mesh = pos.astype(float)/cell_size
    if interlace==1:
        pos_mesh += 0.5
    ###NGP
    if window_order==1:
        imesh = jnp.floor(pos_mesh).astype(np.int32)
    
        imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        imesh = jnp.where( imesh >= 0, imesh, imesh+ng )
        
        field = field.at[imesh[0], imesh[1], imesh[2]].add( weight )
    ###CIC
    elif window_order==2:
        num_neighbors = window_order**3 ### in 3d space
        neighbors = (jnp.arange(num_neighbors, dtype=jnp.int32)[:, jnp.newaxis] // (window_order**jnp.arange(3))) % window_order
        
        imesh = jnp.floor(pos_mesh).astype(np.int32)
        fmesh = jnp.subtract(pos_mesh, imesh)
    
        imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        imesh = jnp.where( imesh >= 0, imesh, imesh+ng )
        
        for neighbor in neighbors:
            neighbor = neighbor.reshape(3, 1, 1, 1)
            neighbor_mesh = imesh + neighbor
            neighbor_mesh &= (ng - 1)   ### periodic in ng
            neighbor_weight = weight * jnp.prod(jnp.where(neighbor > 0, fmesh, 1. - fmesh), axis=0)
            field = field.at[neighbor_mesh[0], neighbor_mesh[1], neighbor_mesh[2]].add(neighbor_weight)

    ###TSC
    elif window_order==3:
        num_neighbors = window_order**3 ### in 3d space
        neighbors = (jnp.arange(num_neighbors, dtype=jnp.int32)[:, jnp.newaxis] // (window_order**jnp.arange(3))) % window_order
        neighbors -= 1   ### the origin should be [0, 0, 0]
        
        imesh = jnp.floor(pos_mesh-1.5).astype(np.int32) + 2
        fmesh = jnp.subtract(pos_mesh, imesh)
                
        imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        imesh = jnp.where( imesh >= 0, imesh, imesh+ng )
                
        for neighbor in neighbors:
            neighbor = neighbor.reshape(3, 1, 1, 1)
            neighbor_mesh = imesh + neighbor
            neighbor_mesh = jnp.where( neighbor_mesh < ng, neighbor_mesh, neighbor_mesh-ng )
            neighbor_mesh = jnp.where( neighbor_mesh >= 0, neighbor_mesh, neighbor_mesh+ng )
            
            w = jnp.prod(jnp.where(neighbor == 0,
                                   0.75 - fmesh*fmesh,
                                   0.5*jnp.where(neighbor == 1,
                                                 fmesh*fmesh + fmesh + 0.25,
                                                 fmesh*fmesh - fmesh + 0.25)),
                                   axis=0)
            field = field.at[neighbor_mesh[0], neighbor_mesh[1], neighbor_mesh[2]].add(w * weight)

    mean = num/ng3
    
    if contd==0:
        field /= mean
    
    return field