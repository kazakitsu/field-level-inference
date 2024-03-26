# !/usr/bin/env python3

import numpy as np
import jax.numpy as jnp
from jax import jit
from functools import partial
import sys
import gc
import jax

jax.config.update("jax_enable_x64", True)

import field_level.coord as coord

@partial(jit, static_argnums=(1, 5, 6, ))
def assign(boxsize, ng, weight, pos, field, window_order, interlace=0, contd=0):
    if len(pos.shape) == 2:
        num = pos.shape[-1]
    elif len(pos.shape) == 4:
        ng_p = pos.shape[-1]
        num = ng_p*ng_p*ng_p
    ng3 = ng*ng*ng
    pos_mesh = pos.astype(float)*ng/boxsize
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
        imesh = jnp.floor(pos_mesh).astype(np.int32)
        fmesh = jnp.subtract(pos_mesh, imesh)
    
        imesh = jnp.where( imesh < ng, imesh, imesh-ng )
        imesh = jnp.where( imesh >= 0, imesh, imesh+ng )
    
        ipmesh = imesh + 1
        ipmesh = jnp.where( ipmesh < ng, ipmesh, ipmesh-ng )
        
        tmesh = jnp.subtract(1.0, fmesh)

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
        imesh = jnp.floor(pos_mesh-1.5).astype(np.int32) + 2
        
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
        field /= mean
    
    return field