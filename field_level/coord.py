# !/usr/bin/env python3

import sys
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from functools import partial


def deconvolve(nvec, window_order):
    ng = nvec.shape[1]
    window = 1.0/jnp.sinc(nvec[0]/ng)/jnp.sinc(nvec[1]/ng)/jnp.sinc(nvec[2]/ng)
    wk = window**window_order
    return wk

def rfftn_kvec(shape, boxsize, dtype=jnp.float32):
    """Generate wavevectors for `jax.numpy.fft.rfftn`"""
    kvec = [jnp.fft.fftfreq(n, d=boxsize/n).astype(dtype) for n in shape[:-1]]
    kvec.append(jnp.fft.rfftfreq(shape[-1], d=boxsize/shape[-1]).astype(dtype))
    kvec = jnp.meshgrid(*kvec, indexing='ij')
    kvec = jnp.stack(kvec, axis=0)
    return kvec * (2 * jnp.pi / boxsize)

def rfftn_khat(shape, dtype=float):
    """unit wavevectors for `jax.numpy.fft.rfftn`
    """
    kvec = [jnp.fft.fftfreq(n, d=1./n).astype(dtype) for n in shape[:-1]]
    kvec.append(jnp.fft.rfftfreq(shape[-1], d=1./shape[-1]).astype(dtype))
    kvec = jnp.meshgrid(*kvec, indexing='ij')
    kvec = jnp.stack(kvec, axis=0)

    kmag = (kvec ** 2).sum(axis=0)
    kmag = jnp.sqrt(kmag)

    kmag = kmag.at[0,0,0].set(1.0)
    kvec = kvec / kmag

    return kvec

def rfftn_k2(kvec):
    """squared wavenumber for `numpy.fft.rfftn`
    """
    k2 = (kvec ** 2).sum(axis=0)
    return k2

def rfftn_disp(kvec):
    k2 = rfftn_k2(kvec)
    k2 = k2.at[0,0,0].set(1.0)
    disp = 1j * kvec / k2
    return disp

def rfftn_nabla(kvec):
    nabla = 1j * kvec
    #nabla = kvec
    return nabla

def rfftn_tide(kvec):
    k2 = rfftn_k2(kvec)
    k2 = k2.at[0,0,0].set(1.0)
    tide = kvec[:, None] * kvec[None] / k2
    trace = jnp.eye(3) / 3.
    trace = jnp.expand_dims(trace, (-1, -2, -3))
    tide -= trace
    tide = tide.at[:,:,0,0,0].set(0.0)
    return tide

def rfftn_G1(kvec):
    k2 = rfftn_k2(kvec)
    k2 = k2.at[0,0,0].set(1.0)
    G1 = jnp.zeros((6,) + k2.shape, dtype=kvec.dtype)
    G1 = G1.at[0].set(kvec[0] * kvec[0] / k2)  # G1_00
    G1 = G1.at[1].set(kvec[0] * kvec[1] / k2)  # G1_01
    G1 = G1.at[2].set(kvec[0] * kvec[2] / k2)  # G1_02
    G1 = G1.at[3].set(kvec[1] * kvec[1] / k2)  # G1_11
    G1 = G1.at[4].set(kvec[1] * kvec[2] / k2)  # G1_12
    G1 = G1.at[5].set(kvec[2] * kvec[2] / k2)  # G1_22
    #G2 = kvec[:, None] * kvec[None] / k2
    #G2 = G2.at[:,:,0,0,0].set(0.0)
    G1 = G1.at[:,0,0,0].set(0.0)
    return G1

def rfftn_Gauss(kvec, R):
    k2 = rfftn_k2(kvec)
    R2 = R*R
    Gauss = jnp.exp(- 0.5 * k2 * R2)
    return Gauss

def coord_func(ng,a,b,c):
    ngo2 = int(ng/2)
    if c > ngo2:
        c = ng - c
    return int((a*ng+b)*(ngo2+1)+c)

def indep_coord(ng):
    ng2 = ng*ng
    ng3 = ng*ng*ng
    ngo2 = int(ng/2)
    ng2o2 = int(ng2/2)
    ng3o2 = int(ng3/2)
    num = ng*ng*(int(ng/2)+1)
    
    ### c = 0 plane
    coord_czero_real = jnp.array([0, coord_func(ng,ngo2,0,0), coord_func(ng,0,ngo2,0), coord_func(ng,ngo2,ngo2,0)])
    
    coord_czero_azero_bhalf = jnp.arange(coord_func(ng,0,1,0), coord_func(ng,0,ngo2-1,0)+1, ngo2+1, dtype=jnp.uint32)
    coord_czero_azero_bhalf_conj = ng2o2 + ng - coord_czero_azero_bhalf

    coord_czero_ahalf_bzero = jnp.arange(coord_func(ng,1,0,0), coord_func(ng,ngo2-1,0,0)+1, ng2o2+ng, dtype=jnp.uint32)
    coord_czero_ahalf_bzero_conj = ng3o2 + ng2 - coord_czero_ahalf_bzero

    coord_czero_ahalf_bngo2 = coord_czero_ahalf_bzero + int(ng2o2/2) + ngo2
    coord_czero_ahalf_bngo2_conj = coord_czero_ahalf_bzero_conj + int(ng2o2/2) + ngo2
    
    coord_czero_aall_bhalf = jnp.array([jnp.arange( coord_func(ng,1,i+1,0), coord_func(ng,ng-1,i+1,0)+1, ng2o2+ng) for i in range(ngo2-1)])
    coord_czero_aall_bhalf_conj = ng3o2 + 3*ng2o2 + ng - coord_czero_aall_bhalf

    ### c = ng/2 plane
    coord_cngo2_real = coord_czero_real + ngo2
    
    coord_cngo2_azero_bhalf = ngo2 + coord_czero_azero_bhalf
    coord_cngo2_azero_bhalf_conj = ngo2 + coord_czero_azero_bhalf_conj

    coord_cngo2_ahalf_bzero = ngo2 + coord_czero_ahalf_bzero
    coord_cngo2_ahalf_bzero_conj = ngo2 + coord_czero_ahalf_bzero_conj

    coord_cngo2_ahalf_bngo2 = ngo2 + coord_czero_ahalf_bngo2
    coord_cngo2_ahalf_bngo2_conj = ngo2 + coord_czero_ahalf_bngo2_conj

    coord_cngo2_aall_bhalf = ngo2 + coord_czero_aall_bhalf
    coord_cngo2_aall_bhalf_conj = ngo2 + coord_czero_aall_bhalf_conj
        
    return coord_czero_real, coord_czero_azero_bhalf, coord_czero_azero_bhalf_conj, \
        coord_czero_ahalf_bzero, coord_czero_ahalf_bzero_conj, \
            coord_czero_ahalf_bngo2, coord_czero_ahalf_bngo2_conj, \
                coord_czero_aall_bhalf, coord_czero_aall_bhalf_conj, \
        coord_cngo2_real, coord_cngo2_azero_bhalf, coord_cngo2_azero_bhalf_conj, \
            coord_cngo2_ahalf_bzero, coord_cngo2_ahalf_bzero_conj, \
                coord_cngo2_ahalf_bngo2, coord_cngo2_ahalf_bngo2_conj, \
                    coord_cngo2_aall_bhalf, coord_cngo2_aall_bhalf_conj

def indep_coord_stack(ng):
    coord_czero_real, coord_czero_azero_bhalf, coord_czero_azero_bhalf_conj, \
        coord_czero_ahalf_bzero, coord_czero_ahalf_bzero_conj, \
            coord_czero_ahalf_bngo2, coord_czero_ahalf_bngo2_conj, \
                coord_czero_aall_bhalf, coord_czero_aall_bhalf_conj, \
    coord_cngo2_real, coord_cngo2_azero_bhalf, coord_cngo2_azero_bhalf_conj, \
        coord_cngo2_ahalf_bzero, coord_cngo2_ahalf_bzero_conj, \
            coord_cngo2_ahalf_bngo2, coord_cngo2_ahalf_bngo2_conj, \
                coord_cngo2_aall_bhalf, coord_cngo2_aall_bhalf_conj = indep_coord(ng)
    
    idx_conjugate_re = jnp.hstack([0,
                                   coord_czero_azero_bhalf_conj,
                                   coord_czero_ahalf_bzero_conj,
                                   coord_czero_ahalf_bngo2_conj,
                                   coord_czero_aall_bhalf_conj.ravel(),
                                   coord_cngo2_azero_bhalf_conj,
                                   coord_cngo2_ahalf_bzero_conj,
                                   coord_cngo2_ahalf_bngo2_conj,
                                   coord_cngo2_aall_bhalf_conj.ravel()])

    idx_conjugate_im = jnp.hstack([coord_czero_real,
                                   coord_cngo2_real,
                                   coord_czero_azero_bhalf_conj,
                                   coord_czero_ahalf_bzero_conj,
                                   coord_czero_ahalf_bngo2_conj,
                                   coord_czero_aall_bhalf_conj.ravel(),
                                   coord_cngo2_azero_bhalf_conj,
                                   coord_cngo2_ahalf_bzero_conj,
                                   coord_cngo2_ahalf_bngo2_conj,
                                   coord_cngo2_aall_bhalf_conj.ravel()])
    
    return idx_conjugate_re, idx_conjugate_im

def kmax_modes(ng, boxsize, kmax):
    k2 = rfftn_k2(rfftn_kvec([ng, ng, ng], boxsize, dtype=float))
    num = k2.size
    return jnp.array(jnp.where(jnp.sqrt(k2.reshape(num)) > kmax))[0]

#@partial(jit, static_argnums=(1, ))
def independent_modes_re_im(deltak_1d, idx_conjugate):
    ### delete conjugate in c=0 & c=ng/2 plane
    deltak_1d = jnp.delete(deltak_1d, jnp.array(idx_conjugate))
    return deltak_1d

#@jit
#@partial(jit, static_argnums=(1, 2))
def independent_modes(deltak, idx_conjugate_real, idx_conjugate_imag):
    num = deltak.size
    deltak_1d = deltak.reshape(num)
    
    deltak_real_1d_ind = independent_modes_re_im(deltak_1d.real, idx_conjugate_real)
    deltak_imag_1d_ind = independent_modes_re_im(deltak_1d.imag, idx_conjugate_imag)
    
    deltak_1d_ind = jnp.hstack([deltak_real_1d_ind, deltak_imag_1d_ind])
    
    return deltak_1d_ind

@partial(jit, static_argnums=(0, ))
def func_reduce(ng_red, array):
    ng = array.shape[0]
    red_idx = jnp.stack([jnp.arange(0,int(ng_red/2)), jnp.arange(ng-int(ng_red/2), ng)],axis=0).flatten()
    
    array_reduced = array[red_idx, :, 0:int(ng_red/2)+1]
    array_reduced = array_reduced[:, red_idx, :]
    
    return array_reduced

@partial(jit, static_argnums=(0,))
def func_extend(ng_ext, array):
    ng = array.shape[0]
    array_extended = (1.+1j)*jnp.zeros((ng_ext, ng_ext, int(ng_ext/2)+1))
    red_idx = jnp.stack([jnp.arange(0,int(ng/2)), jnp.arange(ng_ext-int(ng/2),ng_ext)],axis=0).flatten()
    
    array_extended = array_extended.at[0:int(ng/2), red_idx, 0:int(ng/2)+1].set(array[0:int(ng/2), :,:])
    array_extended = array_extended.at[-int(ng/2):, red_idx, 0:int(ng/2)+1].set(array[-int(ng/2):, :,:])
    
    return array_extended

@partial(jit, static_argnums=(0, ))
def reduce_deltak(ng_red, deltak):
    ng = deltak.shape[0]
    
    coord_czero_real, coord_czero_azero_bhalf, coord_czero_azero_bhalf_conj, \
        coord_czero_ahalf_bzero, coord_czero_ahalf_bzero_conj, \
        coord_czero_ahalf_bngo2, coord_czero_ahalf_bngo2_conj, \
        coord_czero_aall_bhalf, coord_czero_aall_bhalf_conj, \
        coord_cngo2_real, coord_cngo2_azero_bhalf, coord_cngo2_azero_bhalf_conj, \
        coord_cngo2_ahalf_bzero, coord_cngo2_ahalf_bzero_conj, \
        coord_cngo2_ahalf_bngo2, coord_cngo2_ahalf_bngo2_conj, \
        coord_cngo2_aall_bhalf, coord_cngo2_aall_bhalf_conj = indep_coord(ng_red)
    
    ngo2_red = int(ng_red/2)
    num_red = ng_red*ng_red*(int(ng_red/2)+1)

    deltak_red = func_reduce(ng_red, deltak)

    deltak_red_1d = deltak_red.reshape(num_red)
    deltak_red_1d = deltak_red_1d.at[coord_czero_real].set( (deltak_red_1d[coord_czero_real] + deltak_red_1d[coord_czero_real].conj() )/2.)

    deltak_red_1d = deltak_red_1d.at[coord_czero_ahalf_bngo2_conj].set( deltak_red_1d[coord_czero_ahalf_bngo2].conj() )
    deltak_red_1d = deltak_red_1d.at[coord_czero_aall_bhalf_conj[:, ngo2_red-1]].set( deltak_red_1d[coord_czero_aall_bhalf[:, ngo2_red-1]].conj() )

    deltak_red_1d = deltak_red_1d.at[coord_cngo2_real].set( (deltak_red_1d[coord_cngo2_real] + deltak_red_1d[coord_cngo2_real].conj() )/2.)

    deltak_red_1d = deltak_red_1d.at[coord_cngo2_azero_bhalf_conj].set( deltak_red_1d[coord_cngo2_azero_bhalf].conj() )
    deltak_red_1d = deltak_red_1d.at[coord_cngo2_ahalf_bzero_conj].set( deltak_red_1d[coord_cngo2_ahalf_bzero].conj() )

    deltak_red_1d = deltak_red_1d.at[coord_cngo2_ahalf_bngo2_conj].set( deltak_red_1d[coord_cngo2_ahalf_bngo2].conj() )
    deltak_red_1d = deltak_red_1d.at[coord_cngo2_aall_bhalf_conj].set( deltak_red_1d[coord_cngo2_aall_bhalf].conj() )

    deltak_red = deltak_red_1d.reshape(ng_red, ng_red, ngo2_red+1)

    return deltak_red

#@jit
@partial(jit, static_argnums=(1,))
def gauss_1d_to_3d(gaussian_1d, ng):
    ng2 = ng*ng
    ng3 = ng*ng*ng
    ngo2 = int(ng/2)
    ng2o2 = int(ng2/2)
    ng3o2 = int(ng3/2)
    num = ng*ng*(int(ng/2)+1)
    
    coord_czero_real, coord_czero_azero_bhalf, coord_czero_azero_bhalf_conj, \
        coord_czero_ahalf_bzero, coord_czero_ahalf_bzero_conj, \
        coord_czero_ahalf_bngo2, coord_czero_ahalf_bngo2_conj, \
        coord_czero_aall_bhalf, coord_czero_aall_bhalf_conj, \
        coord_cngo2_real, coord_cngo2_azero_bhalf, coord_cngo2_azero_bhalf_conj, \
        coord_cngo2_ahalf_bzero, coord_cngo2_ahalf_bzero_conj, \
        coord_cngo2_ahalf_bngo2, coord_cngo2_ahalf_bngo2_conj, \
        coord_cngo2_aall_bhalf, coord_cngo2_aall_bhalf_conj = indep_coord(ng)
    
    fieldk_re_1d = jnp.zeros(num)
    fieldk_im_1d = jnp.zeros(num)
    
    ### c=0 plane
    fieldk_re_1d = fieldk_re_1d.at[coord_czero_real].set(gaussian_1d[0:4])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_real].set(0.)
    
    fieldk_re_1d = fieldk_re_1d.at[0].set(0.)
    
    fieldk_re_1d = fieldk_re_1d.at[coord_czero_azero_bhalf].set(gaussian_1d[4:ngo2+3])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_azero_bhalf].set(gaussian_1d[ngo2+3:ng+2])
    
    fieldk_re_1d = fieldk_re_1d.at[coord_czero_azero_bhalf_conj].set(fieldk_re_1d[coord_czero_azero_bhalf])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_azero_bhalf_conj].set(-fieldk_im_1d[coord_czero_azero_bhalf])
    
    fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bzero].set(gaussian_1d[ng+2:ng+ngo2+1])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bzero].set(gaussian_1d[ng+ngo2+1:2*ng])

    fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bzero_conj].set(fieldk_re_1d[coord_czero_ahalf_bzero])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bzero_conj].set(-fieldk_im_1d[coord_czero_ahalf_bzero])

    fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bngo2].set(gaussian_1d[2*ng:2*ng+ngo2-1])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bngo2].set(gaussian_1d[2*ng+ngo2-1:3*ng-2])

    fieldk_re_1d = fieldk_re_1d.at[coord_czero_ahalf_bngo2_conj].set(fieldk_re_1d[coord_czero_ahalf_bngo2])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_ahalf_bngo2_conj].set(-fieldk_im_1d[coord_czero_ahalf_bngo2])

    fieldk_re_1d = fieldk_re_1d.at[coord_czero_aall_bhalf.ravel()].set(gaussian_1d[3*ng-2:ng2o2+3*ngo2-1])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_aall_bhalf.ravel()].set(gaussian_1d[ng2o2+3*ngo2-1:ng2])

    fieldk_re_1d = fieldk_re_1d.at[coord_czero_aall_bhalf_conj.ravel()].set(fieldk_re_1d[coord_czero_aall_bhalf.ravel()])
    fieldk_im_1d = fieldk_im_1d.at[coord_czero_aall_bhalf_conj.ravel()].set(-fieldk_im_1d[coord_czero_aall_bhalf.ravel()])
    
    ### c=ng/2 plane
    
    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_real].set(gaussian_1d[ng2:ng2+4])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_real].set(0.)

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_azero_bhalf].set(gaussian_1d[ng2+4:ng2+ngo2+3])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_azero_bhalf].set(gaussian_1d[ng2+ngo2+3:ng2+ng+2])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_azero_bhalf_conj].set(fieldk_re_1d[coord_cngo2_azero_bhalf])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_azero_bhalf_conj].set(-fieldk_im_1d[coord_cngo2_azero_bhalf])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bzero].set(gaussian_1d[ng2+ng+2:ng2+ng+ngo2+1])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bzero].set(gaussian_1d[ng2+ng+ngo2+1:ng2+2*ng])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bzero_conj].set(fieldk_re_1d[coord_cngo2_ahalf_bzero])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bzero_conj].set(-fieldk_im_1d[coord_cngo2_ahalf_bzero])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bngo2].set(gaussian_1d[ng2+2*ng:ng2+2*ng+ngo2-1])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bngo2].set(gaussian_1d[ng2+2*ng+ngo2-1:ng2+3*ng-2])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_ahalf_bngo2_conj].set(fieldk_re_1d[coord_cngo2_ahalf_bngo2])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_ahalf_bngo2_conj].set(-fieldk_im_1d[coord_cngo2_ahalf_bngo2])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_aall_bhalf.ravel()].set(gaussian_1d[ng2+3*ng-2:ng2+ng2o2+3*ngo2-1])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_aall_bhalf.ravel()].set(gaussian_1d[ng2+ng2o2+3*ngo2-1:2*ng2])

    fieldk_re_1d = fieldk_re_1d.at[coord_cngo2_aall_bhalf_conj.ravel()].set(fieldk_re_1d[coord_cngo2_aall_bhalf.ravel()])
    fieldk_im_1d = fieldk_im_1d.at[coord_cngo2_aall_bhalf_conj.ravel()].set(-fieldk_im_1d[coord_cngo2_aall_bhalf.ravel()])
    
    
    fieldk_re_1d = fieldk_re_1d.reshape(ng,ng,ngo2+1).transpose((2,1,0)).reshape(num)
    fieldk_re_1d = fieldk_re_1d.at[ng2:ng3o2].set(gaussian_1d[2*ng2:ng3o2+ng2])
    fieldk_re_1d = fieldk_re_1d.reshape(ngo2+1,ng,ng).transpose((2,1,0)).reshape(num)

    fieldk_im_1d = fieldk_im_1d.reshape(ng,ng,ngo2+1).transpose((2,1,0)).reshape(num)
    fieldk_im_1d = fieldk_im_1d.at[ng2:ng3o2].set(gaussian_1d[ng3o2+ng2:ng3])
    fieldk_im_1d = fieldk_im_1d.reshape(ngo2+1,ng,ng).transpose((2,1,0)).reshape(num)

    #return fieldk_re_1d, fieldk_im_1d
    fieldk_3d =  fieldk_re_1d.reshape(ng,ng,ngo2+1) + 1j*fieldk_im_1d.reshape(ng,ng,ngo2+1)
    return fieldk_3d
