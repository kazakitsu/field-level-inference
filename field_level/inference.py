# !/usr/bin/env python3
import os
import sys
import numpy as np

import jax
import numpyro
if jax.config.read('jax_enable_x64'):
    numpyro.enable_x64()
    print('NumPyro x64 mode is enabled because JAX is in x64 mode.', file=sys.stderr)

import jax.numpy as jnp
from jax import random
import jax.scipy as jsp
from jax import jit
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
from functools import partial
import pickle
import time

import field_level.coord as coord
import field_level.forward_model as forward_model
import field_level.util as utiil
import field_level.cosmo_util as cosmo_util

print('The inference is running on', jax.default_backend(), file=sys.stderr)
cpus = jax.devices("cpu")

def field_inference(boxsize, redshift, which_pk,
                    data_path, save_path,
                    ics_params, model_name, ng_params, mas_params, which_space,
                    cosmo_params, bias_params, err_params, kmax, 
                    dense_mass, mcmc_params,
                    **kwargs):
    """ 
    Field-level inference with numpyro NUTS

    Parameters
    ----------
    boxsize : float
    
    redshift : float
    
    which_pk : str
        The shape of the linear matter power spectrum; now only 'cosmo' is supported.
        
    data_path : ndarray, or str
        The field-level mock data in 3d ndarray or the path to the mock data.
        
    save_path : str
        The base of the save files (mainly chains)
        
    ics_params : (str, int)
        To vary ICs, please set 'varied_ics' as the first element. If the second element is 0, the samples of the ICs are not saved and if 1, they are saved.
        To fix ICs, please specity the path to the true ICs in 3d ndarray with the unit variance (i.e., whose spectrum is white)
        
    model_name : str
        The name of the forward model. Now the following models are supported:
            'gauss' : linear gaussian field in real space
            'gauss_rsd : linear gaussian field in redshift space (i.e., Kaiser formula at the field-level)
            '1lpt' + '_{bias_model}' : Zeldovich displaced bias fields. 'bias_model' can be
                'lin' : the linear Eularian bias; b1
                'quad': the linear & quadratic biases; b1, b2, bG2
                'cubic': the linear, quadratic & effective cubic baises; b1, b2, bG2, bGamma3 from the transfer function
            If you add 'matter' in model_name, the above bias parameters should be regarded as Lagrangian bias
         
    ng_params : (int, int, int)
        The grid sizes for the forwawrd model; (ng, ng_L, ng_E).
        ng : the grid size for ICs
        ng_L : the # of particles. ng_L >= ng
        ng_E ; the grid size of the assignment.
    
    mas_params : (int, int)
        The parameters for the assignment scheme; (window_order, interlace)
        window_order specifies the interpolation scheme; 1 is NGP, 2 is CIC and 3 is TSC
        interlace == 1 enables for the interlacing
    
    which_space : str
        In which space the likelihood is evaluating; 'k_space' or 'r_space'.  ('r-space' is not implemented here though.)
        
    cosmo_params : dict
        The cosmological parameters to sample. The keys should be
            'A' or 'sigma8': the amplitude of the density
            'oc' : the physical density of CDM, oc = Omega_cdm * h^2
            'hubble' : the hubble parameter
    
    bias_params : dict
        The bias parameters to sample, including the counter terms. The keys should be
            'b1': the bias to \delta
            'b2': the bias to \delta^2
            'bG2': the bias to G2
            'bGamma3': the bias to Gamma3
            'c0': the coefficient to k^2 \delta
            'c2' : the coefficient to k^2 \mu^2 \delta
            'c4' : the coefficient to k^2 \mu^4 \delta
            'Sigma2': the coefficient in exp(-0.5 k^2 \Sigma2)
            'Sigma2_mu2' : the coefficient in exp(-0.5 k^2 \mu^2 \Sigma2)

    err_params : dict
        The error (in the likelihood) parameters to sample (or not). The keys should be
            'log_Perr' : The logarithm of the (white) noise power spectrum
            'fixed_log_Perr' : The logarithm of the (white) noise power spectrum, and will not sample it.
    
    kmax : float
        The maximum k used in the likelihood.
        If kmax > 1.0 the cubic cutoff with the kmax^3 grid is used, instead of the default spherical cutoff.
    
    dense_mass : [tuple]
        The parameters whose mass matrix is full-rank.
        
    mcmc_params : (int, int, int, int, int, float, int, int)
        (i-th chain, thinning, # of samples, # of chains, # of warmup, target acceptance rate, random seed for mcmc, # of the previously collected samples (to restart) )
        # of chains can be greater than 1 only if i-th chain < 0.

    """
    which_ics, collect_ics = ics_params
    print(which_ics, file=sys.stderr)
    window_order, interlace = mas_params
    i_chain, thin, n_samples, n_chains, n_warmup, accept_rate, mcmc_seed, i_contd = mcmc_params
    if n_chains > 1 and i_chain >= 0:
        raise ValueError('i_chain must be negative if n_chains > 1')
    if i_contd > 0:
        n_warmup = 1
    if 'fixed_log_Perr' in err_params.keys():
        fixed_log_Perr = err_params.pop('fixed_log_Perr')
        print('fixed_log_Perr = ', fixed_log_Perr, file=sys.stderr)
    if 'fixed_log_Peed' in err_params.keys():
        fixed_log_Peed = err_params.pop('fixed_log_Peed')
        print('fixed_log_Peed = ', fixed_log_Peed, file=sys.stderr)
    if 'fixed_Peded' in err_params.keys():
        fixed_Peded = err_params.pop('fixed_Peded')
        print('fixed_Peded = ', fixed_Peded, file=sys.stderr)

    vol = boxsize*boxsize*boxsize
    
    if which_ics != 'varied_ics':
        print(f'Loading the true initial conditions from {which_ics}...', file=sys.stderr)
        true_gauss_3d = np.load(which_ics)
        print('Done')
       
    if window_order==1:
        w_str = 'ngp'
    elif window_order==2:
        w_str = 'cic'
    elif window_order==3:
        w_str = 'tsc'
    if interlace==1:
        print(f'{w_str} interlacing on', file=sys.stderr)
    elif interlace==0:
        print(f'{w_str} interlacing off', file=sys.stderr)

    if len(ng_params) == 3:
        ng, ng_L, ng_E = ng_params
    elif len(ng_params) == 5:
        ng, ng_L, ng_E, ng_cut, ng_e = ng_params
    print('ng = ', ng, file=sys.stderr)
    print('ng_L = ', ng_L, file=sys.stderr)
    print('ng_E = ', ng_E, file=sys.stderr)
    print('kmax = ', kmax, file=sys.stderr)

    ng3 = ng**3
    ngo2 = int(ng/2)
        
    if 'lpt' in model_name:
        f_model = forward_model.Forward_Model(model_name, which_pk, ng_params, boxsize, which_space, mas_params=mas_params)
    else:
        f_model = forward_model.Forward_Model(model_name, which_pk, ng_params, boxsize, which_space)
        
    ### prepare kvecs
    f_model.kvecs(kmax)
    
    cosmo_params_tmp = [0.02242, 
                        0.11933,
                        0.73,
                        0.9665,
                        3.047,
                        redshift]
    
    ### get the linear power spectrum
    linear_pk = f_model.linear_power(cosmo_params_tmp)
    
    ### find indeces of independent modes
    if which_space == 'k_space':
        if kmax > 1.0:
            kmax = int(kmax)
            ng_max = kmax
            idx_conjugate_real_kmax, idx_conjugate_imag_kmax = coord.indep_coord_stack(ng_max)
            ### before applying this the model must be reduced to be kmax^3
        else:
            idx_conjugate_real, idx_conjugate_imag = coord.indep_coord_stack(ng)
            idx_kmax = coord.kmax_modes(ng, boxsize, kmax)
            idx_conjugate_real_kmax = jnp.unique(jnp.concatenate([idx_conjugate_real, idx_kmax]))
            idx_conjugate_imag_kmax = jnp.unique(jnp.concatenate([idx_conjugate_imag, idx_kmax]))
   
        print('idx_conjugate_real_kmax.shape = ', idx_conjugate_real_kmax.shape, file=sys.stderr)
        print('idx_conjugate_imag_kmax.shape = ', idx_conjugate_imag_kmax.shape, file=sys.stderr)
    elif which_space == 'r_space':
        kmax = int(kmax)
        ng_max = kmax
        ng3_max = ng_max**3
    
    @jit
    def independent_modes(fieldk):
        ###deltak must be ng^3
        fieldk_1d = fieldk.ravel()
    
        fieldk_real_1d_ind = jnp.delete(fieldk_1d.real, idx_conjugate_real_kmax)
        fieldk_imag_1d_ind = jnp.delete(fieldk_1d.imag, idx_conjugate_imag_kmax)
    
        fieldk_1d_ind = jnp.hstack([fieldk_real_1d_ind, fieldk_imag_1d_ind])
    
        return fieldk_1d_ind
    
    ### load a mock data
    if type(data_path) is str:
        print(f'Loading the data from {data_path}...', file=sys.stderr)
        data = np.load(data_path) ### data = signal + noise
        print('Done.', file=sys.stderr)
    else:
        data = data_path
    
    ### Oberved data in 1d independent modes
    if which_space == 'k_space':
        data[0,0,0] = 0.0 ### subtracting the DC mode
        if kmax > 1.0:
            data_max = coord.func_reduce(ng_max, data)
            data_1d_ind = independent_modes(data_max)
        else:
            if ng < ng_E:
                data_E = coord.func_reduce(ng, data)
            elif ng > ng_E:
                data_E = coord.func_extend(ng, data)
            else:
                data_E = data
            data_1d_ind = independent_modes(data_E)
    elif which_space == 'r_space':
        if data.shape[2] != data.shape[0]:
            print('data is in Fourier space.', file=sys.stderr)
            data[0,0,0] = 0.0 ### subtracting the DC mode
            data_ = coord.func_reduce(ng_max, data)
            datar = jnp.fft.irfftn(data_) * ng3_max
            data_1d_ind = datar.reshape(ng3_max)
        else:
            data_1d_ind = data.reshape(ng3_max)
            data_1d_ind -= data_1d_ind.mean()
        print('datar_mean = ', data_1d_ind.mean(), file=sys.stderr)

    print('data_1d_ind.shape = ', file=sys.stderr)
    print(data_1d_ind.shape, file=sys.stderr)
    
    ###k2 in 1d independent modes
    k_NL = 0.1
    if which_space == 'k_space':
        if kmax > 1.0:
            kvec = coord.rfftn_kvec([ng_max,]*3, boxsize)
        else:
            kvec = coord.rfftn_kvec([ng,]*3, boxsize)
        k2 = coord.rfftn_k2(kvec)
        mu2 = kvec[2]*kvec[2]/k2
        k2_1d_ind = independent_modes(k2)/(k_NL*k_NL) ###normalized by (k/k_NL)^2
        mu2_1d_ind = independent_modes(mu2) ###normalized by (k/k_NL)^2
    
    if 'Sigma2' in model_name:
        kvec_E = coord.rfftn_kvec([ng_E,]*3, boxsize)
        k2_E = coord.rfftn_k2(kvec_E)
        mu2_E = kvec_E[2]*kvec_E[2]/k2_E
        del kvec_E

    def model(deltak_data):
        if which_ics=='varied_ics':
            gauss_1d = numpyro.sample("gauss_1d", dist.Normal(0.0, 1.0), sample_shape=(ng3,))
            gauss_3d = coord.gauss_1d_to_3d(gauss_1d, ng)
        
        if 'cosmo' in which_pk:
            if 'oc' in cosmo_params.keys():
                scaled_oc = numpyro.sample('scaled_oc', dist.Uniform(0.0, 1.0))
                #omega_c = numpyro.sample('oc', dist.Uniform(0.05, 0.355))
                omega_c = numpyro.deterministic('oc', 0.05 + (0.355 - 0.05)*scaled_oc)
            else:
                omega_c = 0.11933
            if 'hubble' in cosmo_params.keys():
                scaled_h = numpyro.sample('scaled_hubble', dist.Uniform(0.0, 1.0))
                #hubble = numpyro.sample('hubble', dist.Uniform(0.64, 0.82))
                hubble = numpyro.deterministic('hubble', 0.64 + (0.82 - 0.64)*scaled_h)
                H0 = numpyro.deterministic('H0', hubble*100)
            else:
                hubble = 0.73
            #omega_b = numpyro.sample('ob', dist.Uniform(0.01875, 0.02625))
            omega_b = 0.02242
            OM = numpyro.deterministic('OM', (omega_b + omega_c)/hubble/hubble)
            #ns = numpyro.sample('ns', dist.Uniform(0.84, 1.1))
            ns = 0.9665
            ln1010As = 3.047
            cosmo_params_local = jnp.array([omega_b,
                                            omega_c,
                                            hubble,
                                            ns,
                                            ln1010As,
                                            0.0])  ### z=0.0 for the sigma8 computation
            if 'sigma8' in cosmo_params.keys():
                true_sigma8 = cosmo_params['sigma8']
                scaled_sigma8 = numpyro.sample('scaled_sigma8', dist.Uniform(0.0, 1.0))
                sigma8 = numpyro.deterministic('sigma8', true_sigma8*0.5 + (true_sigma8*1.5 - true_sigma8*0.5)*scaled_sigma8)
                #sigma8 = numpyro.sample('sigma8', dist.Uniform(true_sigma8*0.5, true_sigma8*1.5))
                pk_lin = f_model.linear_power(cosmo_params_local)
                tmp_sigma8 = f_model.sigmaR(pk_lin)
                A = numpyro.deterministic('A', sigma8/tmp_sigma8)
            elif 'A' in cosmo_params.keys():
                A = numpyro.sample('A', dist.Uniform(0.5, 1.5))
                pk_lin = f_model.linear_power(cosmo_params_local)
                sigma8 = numpyro.deterministic('sigma8', A * f_model.sigmaR(pk_lin))
            else:
                A = 1.0
            cosmo_params_local = cosmo_params_local.at[-1].set(redshift)
            if 'cubic' in model_name:
                pk_lin = f_model.linear_power(cosmo_params_local)
        elif 'pow25' in which_pk:
            Pk_kvec = cosmo_util.pow_Pk(jnp.sqrt(k2), 2e4, -2.5)
            Pk_kvec = Pk_kvec.at[0,0,0].set(0.0)
            if 'A' in cosmo_params.keys():
                A = numpyro.sample('A', dist.Uniform(0.5, 1.5))
            else:
                A = 1.0
            
        ### bias
        if 'Ab1' in bias_params.keys():
            true_Ab1 = bias_params['Ab1']
            Ab1 = numpyro.sample('Ab1', dist.Uniform(true_Ab1-1, true_Ab1+2))
            b1 = numpyro.deterministic('b1', Ab1/A)
        elif 'b1' in bias_params.keys():
            true_b1 = bias_params['b1']
            b1 = numpyro.sample('b1', dist.Uniform(true_b1-1, true_b1+2))
        else:
            b1 = 1.0
        
        if 'A2b2' in bias_params.keys():
            true_A2b2 = bias_params['A2b2']
            A2b2 = numpyro.sample('A2b2', dist.Normal(true_A2b2, 2.))
            b2 = numpyro.deterministic('b2', A2b2/A/A)
        elif 'b2' in bias_params.keys():
            true_b2 = bias_params['b2']
            b2 = numpyro.sample('b2', dist.Normal(true_b2, 2.))
        else:
            b2 = -0.5
            
        if 'A2bG2' in bias_params.keys():
            true_A2bG2 = bias_params['A2bG2']
            A2bG2 = numpyro.sample('A2bG2', dist.Normal(true_A2bG2, 2.))
            bG2 = numpyro.deterministic('bG2', A2bG2/A/A)
        elif 'bG2' in bias_params.keys():
            true_bG2 = bias_params['bG2']
            bG2 = numpyro.sample('bG2', dist.Normal(true_bG2, 2.))
        else:
            bG2 = -0.5
            
        if 'A3bGamma3' in bias_params.keys():
            true_A3bGamma3 = bias_params['A3bGamma3']
            A3bGamma3 = numpyro.sample('A2bG2', dist.Normal(true_A3bGamma3, 2.))
            bGamma3 = numpyro.deterministic('bGamma3', dist.Normal(A3bGamma3/A/A/A, 2.))
        elif 'bGamma3' in bias_params.keys():
            true_bGamma3 = bias_params['bGamma3']
            bGamma3 = numpyro.sample('bGamma3', dist.Normal(true_bGamma3, 2.))
        else:
            bGamma3 = -0.5
                    
        ### counter terms
        if 'c0' in bias_params.keys():
            true_c0 = bias_params['c0']
            c0 = numpyro.sample('c0', dist.Normal(true_c0, 20.))
        else:
            #cs2 = -1.2829 or -2.099673 for 1lpt_matter 
            #cs2 = -3.934091 for 1lpt_matter_lin 
            #cs2 = -3.342199 ### for 1lpt_matter_rsd_lin
            cs2 = -2.705611 ### for 1lpt_matter_rsd_quad
        if 'c2' in bias_params.keys():
            true_c2 = bias_params['c2']
            c2 = numpyro.sample('c2', dist.Normal(true_c2, 20.))
        else:
            #c1 = -9.907751 ### for 1lpt_matter_rsd_lin
            c2 = -11.00391 ### for 1lpt_matter_rsd_quad
        if 'c4' in bias_params.keys():
            true_c4 = bias_params['c4']
            c4 = numpyro.sample('c4', dist.Normal(true_c4, 20.))
        else:
            #c2 = 4.668099 ### for 1lpt_matter_rsd_lin
            c4 = 5.498273 ### for 1lpt_matter_rsd_quad
            
        if 'Sigma2' in bias_params.keys():
            true_Sigma2 = bias_params['Sigma2']
            Sigma2 = numpyro.sample('Sigma2', dist.Normal(true_Sigma2, 20.))
        else:
            Sigma2 = 0.0
        if 'Sigma2_mu2' in bias_params.keys():
            true_Sigma2_mu2 = bias_params['Sigma2_mu2']
            Sigma2_mu2 = numpyro.sample('Sigma2_mu2', dist.Normal(true_Sigma2_mu2, 20.))
        else:
            Sigma2_mu2 = 0.0
        
        if 'lin' in model_name:
            biases = [b1,]
        elif 'quad' in model_name:
            biases = [b1, b2, bG2,]
        elif 'cubic' in model_name:
            biases = [b1, b2, bG2, bGamma3,]
        else:
            biases = []
        
        if 'c0' in model_name:
            biases += [c0,]
        if 'c2' in model_name:
            biases += [c2, ]
        if 'c4' in model_name:
            biases += [c4, ]
        if 'Sigma2' in model_name:
            biases += [Sigma2,]
        if 'Sigma2_mu2' in model_name:
            biases += [Sigma2_mu2,]
            
        if 'rsd' in model_name:
            growth_f = numpyro.deterministic('growth_f', cosmo_util.growth_f_fitting(redshift, OM))
            biases += [growth_f]
        
        ### err model
        if 'log_Perr' in err_params.keys(): 
            #true_log_Perr = jnp.log(true_Perr)
            if err_params['log_Perr'] > 15.:
                true_log_Perr = jnp.log((vol/err_params['log_Perr'])**3)
            else:
                true_log_Perr = err_params['log_Perr']
            log_Perr = numpyro.sample("log_Perr", dist.Normal(true_log_Perr, 0.5))
            Perr = jnp.exp(log_Perr)
            #ratio_err = Perr/true_Perr
            #Pres = numpyro.deterministic("Pres", ratio_err - 1.0)
        else:
            if fixed_log_Perr > 15.:
                Perr = (vol/fixed_log_Perr)**3
            else:
                Perr = jnp.exp(fixed_log_Perr)
        if 'log_Perr_k2mu2' in err_params.keys():
            true_log_Perr_k2mu2 = err_params['log_Perr_k2mu2']
            log_Perr_k2mu2 = numpyro.sample("log_Perr_k2mu2", dist.Normal(true_log_Perr_k2mu2, 0.5))
            Perr_k2mu2 = jnp.exp(log_Perr_k2mu2)
            ratio_ani = numpyro.deterministic("ratio_ani", Perr_k2mu2/Perr)
            Perr = Perr + Perr*Perr_k2mu2*k2_1d_ind*mu2_1d_ind/(k_NL*k_NL)
            
        if 'Perr_k2'in err_params.keys():
            true_Perr_k2 = err_params['Perr_k2']
            Perr_k2 = numpyro.sample("Perr_k2", dist.Normal(0., 20.0))
            Perr = Perr + Perr*Perr_k2*k2_1d_ind
        
        if which_space == 'k_space':
            sigma_err = jnp.sqrt(Perr/(2.*vol))
        elif which_space == 'r_space':
            sigma2_err = Perr*ng3_max/vol

        ### for the density-dependent noise
        if 'log_Peded' in err_params.keys():
            true_log_Peded = err_params['log_Peded']
            log_Peded = numpyro.sample("log_Peded", dist.Normal(true_log_Peded, 0.5))
            Peded = numpyro.deterministic("Peded", jnp.exp(log_Peded))
            bound = jnp.sqrt(Perr*Peded)
            if which_space == 'k_space':
                sigma_eded = jnp.sqrt(Peded/(2.*vol))
            elif which_space == 'r_space':
                sigma2_eded = Peded*ng3_max/vol
        elif 'fixed_log_Peded' in err_params.keys():
            fixed_log_Peded = err_params['fixed_log_Peded']
            Peded = jnp.exp(fixed_log_Peded)
            if which_space == 'k_space':
                sigma_eded = jnp.sqrt(Peded/(2.*vol))
            elif which_space == 'r_space':
                sigma2_eded = Peded*ng3_max/vol
        if 'Peed' in err_params.keys():
            scaled_Peed = numpyro.sample("scaled_Peed", dist.Uniform(-1, 1.))
            Peed = numpyro.deterministic("Peed", scaled_Peed*bound)
            ratio = numpyro.deterministic("ratio", Peed/Peded)
            if which_space == 'k_space':
                Perr_eff = numpyro.deterministic("Perr_eff", Perr - Peed*Peed/Peded)
                sigma_err = jnp.sqrt(Perr_eff/(2.*vol))
            elif which_space == 'r_space':
                sigma2_eed = Peed*ng3_max/vol
        elif 'fixed_Peed' in err_params.keys():
            Peed = jnp.exp(fixed_log_Peed)
            if which_space == 'k_space':
                Perr_eff = numpyro.deterministic("Perr_eff", Perr - Peed*Peed/Peded)
                sigma_err = jnp.sqrt(Perr_eff/(2.*vol))
            elif which_space == 'r_space':
                sigma2_eed = Peed*ng3_max/vol
            
        ### construct the linear field
        if which_ics=='varied_ics':
            delk = A * f_model.linear_modes(cosmo_params_local, gauss_3d)
        else:
            delk = A * f_model.linear_modes(cosmo_params_local, true_gauss_3d)

        delk_L = coord.func_extend(ng_L, delk)
                
        if which_space == 'k_space':
            if 'cubic' in model_name:
                fieldk_model_E = f_model.models(delk_L, biases, pk_lin)
            else:
                fieldk_model_E = f_model.models(delk_L, biases)
                                    
            if 'Sigma2_mu2' in model_name:
                fieldk_model_E = fieldk_model_E * jnp.exp(-0.5*Sigma2_mu2*k2_E*mu2_E)
            if 'Sigma2' in model_name:
                fieldk_model_E = fieldk_model_E * jnp.exp(-0.5*Sigma2*k2_E)

            if kmax > 1.0:
                fieldk_model = coord.func_reduce(ng_max, fieldk_model_E)
            else:
                if ng < ng_E:
                    fieldk_model = coord.func_reduce(ng, fieldk_model_E)
                elif ng == ng_E:
                    fieldk_model = fieldk_model_E
                elif ng > ng_E:
                    fieldk_model = coord.func_extend(ng, fieldk_model_E)
            field_model_1d_ind = independent_modes(fieldk_model)
        elif which_space == 'r_space':
            if 'cubic' in model_name:
                fieldr_model, delr_max, d2r_max = f_model.models(delk_L, biases, pk_lin)
            else:
                fieldr_model, delr_max, d2r_max = f_model.models(delk_L, biases)
            field_model_1d_ind = fieldr_model.reshape(ng3_max)
            if 'log_Peded' in err_params.keys() or 'fixed_log_Peded' in err_params.keys():
                d2r_1d_ind = d2r_max.reshape(ng3_max)
            if 'Peed' in err_params.keys() or 'fixed_Peed' in err_params.keys():
                delr_1d_ind = delr_max.reshape(ng3_max)
            if 'Peded' in err_params.keys() or 'fixed_Peded' in err_params.keys() or 'Peed' in err_params.keys() or 'fixed_Peed' in err_params.keys():
                sigma2_err = sigma2_err + 2.0*sigma2_eed*delr_1d_ind + sigma2_eded*d2r_1d_ind
            sigma_err = jnp.sqrt(sigma2_err)

        Y = numpyro.sample('Y', dist.Normal(field_model_1d_ind, sigma_err), obs=deltak_data)
    '''
    params = []

    params += list(cosmo_params.keys())
    if 'A' in cosmo_params.keys():
        params += ['sigma8']
    if 'sigma8' in cosmo_params.keys():
        params += ['A']
    if 'hubble' in cosmo_params.keys():
        params += ['H0']
    if 'oc' in cosmo_params.keys() or 'hubble' in cosmo_params.keys():
        params += ['OM']
    if 'rsd' in model_name:
        params += ['growth_f',]

    params += list(bias_params.keys())
    params += list(err_params.keys())
    if 'log_Peded' in err_params.keys():
        params += ['Peded']
    if 'Peed' in err_params.keys():
        params += ['scaled_Peed']
        params += ['ratio']
    
    if collect_ics==1:
        params += ['gauss_1d']

    min_pe_params = params.copy()
    if which_ics=='varied_ics':
        min_pe_params += ['gauss_1d']
    '''
   # print('save params = ', params, file=sys.stderr)
   # print('min_pe_params = ', min_pe_params, file=sys.stderr)
    print('dense_mass = ', dense_mass, file=sys.stderr)
    
    n_total = thin*n_samples
    i_sample = 100
    i_iter = int(n_total/i_sample)

    kernel = numpyro.infer.NUTS(model=model,
                                     target_accept_prob=accept_rate,
                                     adapt_step_size=True,
                                     adapt_mass_matrix=True,
                                     dense_mass=dense_mass,
                                     max_tree_depth=(9, 9),
                                     init_strategy=numpyro.infer.init_to_sample)

    mcmc = numpyro.infer.MCMC(kernel,
                              num_samples=i_sample,
                              num_warmup=n_warmup,
                              num_chains=n_chains,
                              thinning=thin,
                              #chain_method="sequential",
                              chain_method="parallel",
                              progress_bar=False)
    
    posterior_samples = {}
    min_pe_samples = {}
                        
    save_base = f'{save_path}'
    print('save_base = ', save_base, file=sys.stderr)

    if i_contd > 0:
        #samples_previous = np.loadtxt(f'{save_base}_{params[0]}_chain{i_chain}.dat')
        #print(f'{params[0]} samples_previous.shape = ', samples_previous.shape, file=sys.stderr)
        samples_previous = np.loadtxt(f'{save_base}_A_chain{i_chain}.dat')
        print(f'A samples_previous.shape = ', samples_previous.shape, file=sys.stderr)
        i_contd_check = samples_previous.shape[0] // i_sample
        print('i_contd = ', i_contd, file=sys.stderr)
        if i_contd != i_contd_check:
            print('i_contd != number of samples_previous * i_sample', file=sys.stderr)
            sys.exit(1)
        else:
            print('i_contd == number of samples_previous * i_sample', file=sys.stderr)

    if i_contd > 0:
        rng_key = jax.random.PRNGKey(0)
        mcmc.warmup(rng_key, deltak_data=data_1d_ind, extra_fields=('potential_energy',))
        print('Restarting samping...', file=sys.stderr)
        with open(f'{save_base}_{i_chain}_{i_contd}_last_state.pkl', 'rb') as f:
            last_state = pickle.load(f)
        mcmc.post_warmup_state = last_state
        mcmc._last_state = last_state
        print('LOADED last state = ', file=sys.stderr)
        print(mcmc.last_state, file=sys.stderr)
        samples = mcmc.get_samples()
        params = list(samples.keys())
        min_pe_params = params.copy()
        if collect_ics == 0:
            params.remove('gauss_1d')
        min_pe = np.loadtxt(f'{save_base}_min_pe_chain{i_chain}.dat')
        for param in params:
            min_pe_samples[param] = np.loadtxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat')
    elif n_warmup == 1 and (i_chain == 1 or i_chain == 2):
    #elif i_chain == 0:
    #elif i_chain > 0:
        rng_key = jax.random.PRNGKey(0)
        n_warmup = 1
        mcmc.warmup(rng_key, deltak_data=data_1d_ind, extra_fields=('potential_energy',))
        #print(f'Loading post warmup state from {save_base}_2_24710_warmup_state.pkl ...', file=sys.stderr)
        #print(f'Loading post warmup state from {save_base}_1_24710_warmup_state.pkl ...', file=sys.stderr)
        #print(f'Loading post warmup state from {save_base}_1_12355_warmup_state.pkl ...', file=sys.stderr)
        print(f'Loading post warmup state from {save_base}_0_{mcmc_seed}_warmup_state.pkl ...', file=sys.stderr)
        #with open(f'{save_base}_2_24710_warmup_state.pkl', 'rb') as f:
        with open(f'{save_base}_0_{mcmc_seed}_warmup_state.pkl', 'rb') as f:
        #with open(f'{save_base}_1_12355_warmup_state.pkl', 'rb') as f:
            warmup_state = pickle.load(f)
        mcmc.post_warmup_state = warmup_state
        mcmc._last_state = warmup_state
        print('LOADED warmup state = ', file=sys.stderr)
        print(mcmc.post_warmup_state, file=sys.stderr)
    elif n_warmup == 1:
        mcmc_seed += 12345*i_chain
        rng_key = jax.random.PRNGKey(0)
        mcmc.warmup(rng_key, deltak_data=data_1d_ind, extra_fields=('potential_energy',))
        print(f'Loading post warmup state from {save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl ...', file=sys.stderr)
        with open(f'{save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl', 'rb') as f:
            warmup_state = pickle.load(f)
        mcmc.post_warmup_state = warmup_state
        mcmc._last_state = warmup_state
        print('LOADED warmup state = ', file=sys.stderr)
        print(mcmc.post_warmup_state, file=sys.stderr)
        print('LOADED last state = ', file=sys.stderr)
        print(mcmc.last_state, file=sys.stderr)
    else:
        mcmc_seed += 12345*i_chain
        rng_key = jax.random.PRNGKey(mcmc_seed)
        print('rng_seed = ', mcmc_seed, file=sys.stderr)
        mcmc.warmup(rng_key, deltak_data=data_1d_ind, extra_fields=('potential_energy',), collect_warmup=False)
        inv_mass_matrix = mcmc.post_warmup_state.adapt_state.inverse_mass_matrix
    
        i_warmup = 1
        print(inv_mass_matrix, file=sys.stderr)
        print('i_warmup = ', i_warmup, file=sys.stderr)
       
        criteria = 0.9
        
        while_check = 0
        
        if 'A' in dense_mass[0]:
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][0,0]) > criteria)
        
        if 'A' in dense_mass[0] and 'scaled_oc' in dense_mass[0] and 'scaled_hubble' in dense_mass[0]:
            num_cosmo_params = 3
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][1,1]) > criteria)
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][2,2]) > criteria)
        elif 'A' in dense_mass[0] and 'scaled_oc' in dense_mass[0]:
            num_cosmo_params = 2
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][1,1]) > criteria)
        elif 'scaled_oc' in dense_mass[0] and 'scaled_hubble' in dense_mass[0]:
            num_cosmo_params = 2
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][1,1]) > criteria)
        elif 'scaled_hubble' in dense_mass[0] and 'A' in dense_mass[0]:
            num_cosmo_params = 2
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][1,1]) > criteria)
        elif 'A' in dense_mass[0] or 'scaled_oc' in dense_mass[0] or 'scaled_hubble' in dense_mass[0]:
            num_cosmo_params = 1
        else:
            num_cosmo_params = 0
            
        num_bias_params = 0
        if 'b1' in dense_mass[0] or 'Ab1' in dense_mass[0]:
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][num_cosmo_params,num_cosmo_params]) > criteria)
            print('idx of dense mass @ b1 = ', num_cosmo_params, file=sys.stderr)
            print('dense mass @ b1 = ', inv_mass_matrix[dense_mass[0]][num_cosmo_params,num_cosmo_params], file=sys.stderr)
            num_bias_params += 1
        if 'b2' in dense_mass[0] or 'A2b2' in dense_mass[0]:
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params]) > criteria)
            print('idx of dense mass @ b2 = ', num_cosmo_params+num_bias_params, file=sys.stderr)
            print('dense mass @ b2 = ', inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params], file=sys.stderr)
            num_bias_params += 1
        if 'bG2' in dense_mass[0] or 'A2bG2' in dense_mass[0]:
            while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params]) > criteria)
            print('idx of dense mass @ bG2 = ', num_cosmo_params+num_bias_params, file=sys.stderr)
            print('dense mass @ bG2 = ', inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params], file=sys.stderr)

        print('while = ', while_check, file=sys.stderr)
        
        with open(f'{save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl', 'wb') as f:
            pickle.dump(mcmc.post_warmup_state, f)
        
        while( while_check > 0 ):
            rng_key = jax.random.PRNGKey(mcmc_seed+i_warmup)
            print('rng_seed = ',mcmc_seed+i_warmup, file=sys.stderr)
            print('rng key = ', rng_key, file=sys.stderr)
            mcmc.warmup(rng_key, deltak_data=data_1d_ind, extra_fields=('potential_energy',))
            inv_mass_matrix = mcmc.post_warmup_state.adapt_state.inverse_mass_matrix
            i_warmup += 1
            print(inv_mass_matrix, file=sys.stderr)
            
            while_check = 0
        
            if 'A' in dense_mass[0]:
                while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][0,0]) > criteria)
        
            if 'A' in dense_mass[0] and 'scaled_oc' in dense_mass[0] and 'scaled_hubble' in dense_mass[0]:
                num_cosmo_params = 3
            elif 'A' in dense_mass[0] and 'scaled_oc' in dense_mass[0]:
                num_cosmo_params = 2
            elif 'scaled_oc' in dense_mass[0] and 'scaled_hubble' in dense_mass[0]:
                num_cosmo_params = 2
            elif 'scaled_hubble' in dense_mass[0] and 'A' in dense_mass[0]:
                num_cosmo_params = 2
            elif 'A' in dense_mass[0] or 'scaled_oc' in dense_mass[0] or 'scaled_hubble' in dense_mass[0]:
                num_cosmo_params = 1
            else:
                num_cosmo_params = 0
            
            num_bias_params = 0
            if 'b1' in dense_mass[0]:
                while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][num_cosmo_params,num_cosmo_params]) > criteria)
                print('idx of dense mass @ b1 = ', num_cosmo_params, file=sys.stderr)
                print('dense mass @ b1 = ', inv_mass_matrix[dense_mass[0]][num_cosmo_params,num_cosmo_params], file=sys.stderr)
                num_bias_params += 1
            if 'b2' in dense_mass[0]:
                while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params]) > criteria)
                print('idx of dense mass @ b2 = ', num_cosmo_params+num_bias_params, file=sys.stderr)
                print('dense mass @ b2 = ', inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params], file=sys.stderr)
                num_bias_params += 1
            if 'bG2' in dense_mass[0]:
                while_check += (jnp.abs(inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params]) > criteria)
                print('idx of dense mass @ bG2 = ', num_cosmo_params+num_bias_params, file=sys.stderr)
                print('dense mass @ bG2 = ', inv_mass_matrix[dense_mass[0]][num_cosmo_params+num_bias_params,num_cosmo_params+num_bias_params], file=sys.stderr)

            print('while = ', while_check, file=sys.stderr)
            
            print('i_warmup = ', i_warmup, file=sys.stderr)
        
            with open(f'{save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl', 'wb') as f:
                pickle.dump(mcmc.post_warmup_state, f)

    for i in range(i_iter):
        print(f'running batch {i} ...', file=sys.stderr)
        rng_key = jax.random.PRNGKey(mcmc_seed+12*i_chain+123*i+1234*i_contd)
        mcmc.run(rng_key, deltak_data=data_1d_ind, extra_fields=('potential_energy',))
        samples = mcmc.get_samples()
        params = list(samples.keys())
        min_pe_params = params.copy()
        if collect_ics == 0:
            params.remove('gauss_1d')
        pes = mcmc.get_extra_fields()['potential_energy']
        min_pe_idx = jnp.argmin(pes)
        if i == 0:
            if i_contd==0:
                min_pe = pes[min_pe_idx]
                print('Min of the potential energy = ', min_pe, file=sys.stderr)
                for param in min_pe_params:
                    min_pe_samples[param] = samples[param][min_pe_idx]
                    print(f'min_pe_samples[{param}] = ', min_pe_samples[param], file=sys.stderr)
            else:
                if pes[min_pe_idx] < min_pe:
                    min_pe = pes[min_pe_idx]
                    print('Min of the potential energy = ', min_pe, file=sys.stderr)
                    for param in min_pe_params:
                        min_pe_samples[param] = samples[param][min_pe_idx]
                        print(f'min_pe_samples[{param}] = ', min_pe_samples[param], file=sys.stderr)
            if which_ics=='varied_ics':
                mean_gauss_1d = jnp.mean(samples['gauss_1d'], axis=0)
            for param in params:
                posterior_samples[param] = jax.device_put(samples[param].astype(np.float32), cpus[0])
            posterior_samples['potential_energy'] = jax.device_put(pes.astype(np.float32), cpus[0])
        else:
            if pes[min_pe_idx] < min_pe:
                min_pe = pes[min_pe_idx]
                print('Min of the potential energy = ', min_pe, file=sys.stderr)
                for param in min_pe_params:
                    min_pe_samples[param] = samples[param][min_pe_idx]
                    print(f'min_pe_samples[{param}] = ', min_pe_samples[param], file=sys.stderr)
            if which_ics=='varied_ics':
                tmp_mean = jnp.mean(samples['gauss_1d'], axis=0)
                mean_gauss_1d = ( i*mean_gauss_1d + tmp_mean )/(i+1)
            for param in params:
                posterior_samples[param] = np.concatenate([posterior_samples[param], samples[param].astype(np.float32)])
            posterior_samples['potential_energy'] = np.concatenate([posterior_samples['potential_energy'], pes.astype(np.float32)])
        del samples
        mcmc.post_warmup_state = mcmc.last_state
        if(i==0):
            print(f'i={i}, mcmc.last_state = ', file=sys.stderr)
            print(mcmc.last_state, file=sys.stderr)
        elif(i==i_iter-1):
            print(f'i={i}, mcmc.last_state = ', file=sys.stderr)
            print(mcmc.last_state, file=sys.stderr)
            i = i + i_contd + 1
            with open(f'{save_base}_{i_chain}_{i}_last_state.pkl', 'wb') as f:
                pickle.dump(mcmc.last_state, f)
        #gc.collect()
    
    params.append('potential_energy')

    for param in params:
        print(f'posterior_sample[{param}]', posterior_samples[param].shape, file=sys.stderr)
        print(f'posterior_sample[{param}] = ', posterior_samples[param], file=sys.stderr)
        if(i_contd>0):
            samples_previous = np.loadtxt(f'{save_base}_{param}_chain{i_chain}.dat')
            print('samples_previous.shape = ', samples_previous.shape, file=sys.stderr)
            if param=='gauss_1d':
                posterior_samples[param] = np.vstack([samples_previous.astype(np.float32), posterior_samples[param]])
            elif param=='gauss_1d_e':
                posterior_samples[param] = np.vstack([samples_previous.astype(np.float32), posterior_samples[param]])
            else:
                posterior_samples[param] = np.hstack([samples_previous.astype(np.float32), posterior_samples[param]])
                posterior_samples[param] = posterior_samples[param].reshape(-1,1)
        np.savetxt(f'{save_base}_{param}_chain{i_chain}.dat', posterior_samples[param])

    if which_ics=='varied_ics':
        if(i_contd>0):
            mean_gauss_1d_previous = np.loadtxt(f'{save_base}_gauss_1d_mean_chain{i_chain}.dat')
            mean_gauss_1d = ( n_samples*mean_gauss_1d + i_contd*int(i_sample/thin)*mean_gauss_1d_previous )/(n_samples + i_contd*int(i_sample/thin))
        np.savetxt(f'{save_base}_gauss_1d_mean_chain{i_chain}.dat', mean_gauss_1d)

    np.savetxt(f'{save_base}_min_pe_chain{i_chain}.dat', np.array([min_pe]))
    for param in min_pe_params:
        print(f'min_pe_sample[{param}]', min_pe_samples[param].shape, file=sys.stderr)
        if param=='gauss_1d':
            np.savetxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat', min_pe_samples[param])
        elif param=='gauss_1d_e':
            np.savetxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat', min_pe_samples[param])
        else:
            np.savetxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat', np.array([min_pe_samples[param]]))

    if i_contd > 0:
        os.remove(f'{save_base}_{i_chain}_{i_contd}_last_state.pkl')
    #if i_chain > 0:
    #    os.remove(f'{save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl')
