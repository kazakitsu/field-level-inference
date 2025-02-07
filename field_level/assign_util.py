# !/usr/bin/env python3

import sys
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial

import field_level.coord as coord

#def assign(boxsize, ng, weight, pos, window_order, interlace=0, contd=0):
@partial(jit, static_argnums=(4, 5, 6))
def assign(boxsize, field, weight, pos, window_order, interlace=0, contd=0):
    #field = jnp.zeros((ng, ng, ng))
    ng = field.shape[0]
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
    elif window_order == 2:
        ### somehow the following commented code is not working for the inference with the quad bias operatiors...
        '''
        num_neighbors = window_order**3 ### in 3d space
        neighbors = (jnp.arange(num_neighbors, dtype=jnp.int32)[:, jnp.newaxis] // (window_order**jnp.arange(3))) % window_order
        
        imesh = jnp.floor(pos_mesh).astype(np.int32)
        fmesh = jnp.subtract(pos_mesh, imesh)
    
        imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        imesh = jnp.where( imesh >= 0, imesh, imesh+ng )
        
        for neighbor in neighbors:
            if len(pos.shape) == 2:
                neighbor = neighbor.reshape(3, 1)
            elif len(pos.shape) == 4:
                neighbor = neighbor.reshape(3, 1, 1, 1)
            neighbor_mesh = imesh + neighbor
            neighbor_mesh &= (ng - 1)   ### periodic in ng
            neighbor_weight = weight * jnp.prod(jnp.where(neighbor > 0, fmesh, 1. - fmesh), axis=0)
            field = field.at[neighbor_mesh[0], neighbor_mesh[1], neighbor_mesh[2]].add(neighbor_weight)
        '''
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = pos_mesh - imesh

        #imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        #imesh = jnp.where( imesh >= 0, imesh, imesh+ng )

        #ipmesh = imesh + 1
        #ipmesh = jnp.where( ipmesh < ng, ipmesh, ipmesh-ng )

        #tmesh = jnp.subtract(1.0, fmesh)

        imesh = imesh % ng
        ipmesh = (imesh + 1) % ng

        tmesh = 1.0 - fmesh
        
        field = field.at[imesh[0], imesh[1], imesh[2]].add( weight*tmesh[0]*tmesh[1]*tmesh[2] )

        field = field.at[ipmesh[0], imesh[1], imesh[2]].add( weight*fmesh[0]*tmesh[1]*tmesh[2] )
        field = field.at[imesh[0], ipmesh[1], imesh[2]].add( weight*tmesh[0]*fmesh[1]*tmesh[2] )
        field = field.at[imesh[0], imesh[1], ipmesh[2]].add( weight*tmesh[0]*tmesh[1]*fmesh[2] )

        field = field.at[ipmesh[0], ipmesh[1], imesh[2]].add( weight*fmesh[0]*fmesh[1]*tmesh[2] )
        field = field.at[imesh[0], ipmesh[1], ipmesh[2]].add( weight*tmesh[0]*fmesh[1]*fmesh[2] )
        field = field.at[ipmesh[0], imesh[1], ipmesh[2]].add( weight*fmesh[0]*tmesh[1]*fmesh[2] )

        field = field.at[ipmesh[0], ipmesh[1], ipmesh[2]].add( weight*fmesh[0]*fmesh[1]*fmesh[2] )

    ###TSC
    elif window_order==3:
        ### somehow the following commented code is not working for the inference with the quad bias operatiors...
        '''
        num_neighbors = window_order**3 ### in 3d space
        neighbors = (jnp.arange(num_neighbors, dtype=jnp.int32)[:, jnp.newaxis] // (window_order**jnp.arange(3))) % window_order
        neighbors -= 1   ### the origin should be [0, 0, 0]
        
        imesh = jnp.floor(pos_mesh-1.5).astype(np.int32) + 2
        fmesh = jnp.subtract(pos_mesh, imesh)
                
        imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        imesh = jnp.where( imesh >= 0, imesh, imesh+ng )
                
        for neighbor in neighbors:
            if len(pos.shape) == 2:
                neighbor = neighbor.reshape(3, 1)
            elif len(pos.shape) == 4:
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
        '''
        imesh = jnp.floor(pos_mesh-1.5).astype(jnp.int32) + 2
        fmesh = jnp.subtract(pos_mesh, imesh)

        imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        imesh = jnp.where( imesh >= 0, imesh, imesh+ng )

        ipmesh = imesh + 1
        ipmesh = jnp.where( ipmesh < ng, ipmesh, ipmesh-ng )

        immesh = imesh - 1
        immesh = jnp.where( immesh >= 0, immesh, immesh+ng )        

        tmesh = 0.75 - fmesh*fmesh

        pmesh = 0.5*(fmesh*fmesh + fmesh + 0.25)
        mmesh = 0.5*(fmesh*fmesh - fmesh + 0.25)

        field = field.at[imesh[0], imesh[1], imesh[2]].add( weight*tmesh[0]*tmesh[1]*tmesh[2] )

        field = field.at[ipmesh[0], imesh[1], imesh[2]].add( weight*pmesh[0]*tmesh[1]*tmesh[2] )
        field = field.at[immesh[0], imesh[1], imesh[2]].add( weight*mmesh[0]*tmesh[1]*tmesh[2] )
        field = field.at[imesh[0], ipmesh[1], imesh[2]].add( weight*tmesh[0]*pmesh[1]*tmesh[2] )
        field = field.at[imesh[0], immesh[1], imesh[2]].add( weight*tmesh[0]*mmesh[1]*tmesh[2] )
        field = field.at[imesh[0], imesh[1], ipmesh[2]].add( weight*tmesh[0]*tmesh[1]*pmesh[2] )
        field = field.at[imesh[0], imesh[1], immesh[2]].add( weight*tmesh[0]*tmesh[1]*mmesh[2] )

        field = field.at[ipmesh[0], ipmesh[1], imesh[2]].add( weight*pmesh[0]*pmesh[1]*tmesh[2] )
        field = field.at[ipmesh[0], immesh[1], imesh[2]].add( weight*pmesh[0]*mmesh[1]*tmesh[2] )
        field = field.at[immesh[0], ipmesh[1], imesh[2]].add( weight*mmesh[0]*pmesh[1]*tmesh[2] )
        field = field.at[immesh[0], immesh[1], imesh[2]].add( weight*mmesh[0]*mmesh[1]*tmesh[2] )

        field = field.at[ipmesh[0], imesh[1], ipmesh[2]].add( weight*pmesh[0]*tmesh[1]*pmesh[2] )
        field = field.at[ipmesh[0], imesh[1], immesh[2]].add( weight*pmesh[0]*tmesh[1]*mmesh[2] )
        field = field.at[immesh[0], imesh[1], ipmesh[2]].add( weight*mmesh[0]*tmesh[1]*pmesh[2] )
        field = field.at[immesh[0], imesh[1], immesh[2]].add( weight*mmesh[0]*tmesh[1]*mmesh[2] )

        field = field.at[imesh[0], ipmesh[1], ipmesh[2]].add( weight*tmesh[0]*pmesh[1]*pmesh[2] )
        field = field.at[imesh[0], ipmesh[1], immesh[2]].add( weight*tmesh[0]*pmesh[1]*mmesh[2] )
        field = field.at[imesh[0], immesh[1], ipmesh[2]].add( weight*tmesh[0]*mmesh[1]*pmesh[2] )
        field = field.at[imesh[0], immesh[1], immesh[2]].add( weight*tmesh[0]*mmesh[1]*mmesh[2] )

        field = field.at[ipmesh[0], ipmesh[1], ipmesh[2]].add( weight*pmesh[0]*pmesh[1]*pmesh[2] )
        field = field.at[ipmesh[0], ipmesh[1], immesh[2]].add( weight*pmesh[0]*pmesh[1]*mmesh[2] )
        field = field.at[ipmesh[0], immesh[1], ipmesh[2]].add( weight*pmesh[0]*mmesh[1]*pmesh[2] )
        field = field.at[ipmesh[0], immesh[1], immesh[2]].add( weight*pmesh[0]*mmesh[1]*mmesh[2] )

        field = field.at[immesh[0], ipmesh[1], ipmesh[2]].add( weight*mmesh[0]*pmesh[1]*pmesh[2] )
        field = field.at[immesh[0], ipmesh[1], immesh[2]].add( weight*mmesh[0]*pmesh[1]*mmesh[2] )
        field = field.at[immesh[0], immesh[1], ipmesh[2]].add( weight*mmesh[0]*mmesh[1]*pmesh[2] )
        field = field.at[immesh[0], immesh[1], immesh[2]].add( weight*mmesh[0]*mmesh[1]*mmesh[2] )
        
        
    mean = num/ng3
    
    if contd==0:
        field = field / mean
    
    return field
