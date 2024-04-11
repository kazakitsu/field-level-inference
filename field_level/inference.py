# !/usr/bin/env python3
import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS
import jax
import jax.scipy as jsp
from jax import jit
from functools import partial
import os
import sys
import matplotlib.pyplot as plt
import pickle
import time

from field_level.JAX_Zenbu import Zenbu
from field_level.Zenbu_utils.loginterp_jax import loginterp_jax

import field_level.coord as coord
import field_level.forward_model as forward_model
import field_level.util as utiil
import field_level.cosmo_util as cosmo_util

import jax
jax.config.update("jax_enable_x64", True)

numpyro.enable_x64()
print('The inference is running on', jax.default_backend(), file=sys.stderr)
cpus = jax.devices("cpu")

def field_inference(boxsize, redshift, which_pk,
                    data_path, save_path,
                    ics_params, model_name, ng_params, mas_params, which_space,
                    cosmo_params, bias_params, err_params, kmax, 
                    dense_mass, mcmc_params,
                    **kwargs):
    """ Field-level inference with numpyro NUTS

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
            'cs2': the coefficient to k^2 \delta
            'c1' : the coefficient to k^2 \mu^2 \delta
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
        
    mcmc_params : (int, int, int, int, float, int, int)
        (i-th chain, thinning, # of samples, # of warmup, target acceptance rate, random seed for mcmc, # of the previously collected samples (to restart) )

    """
    which_ics, collect_ics = ics_params
    print(which_ics, file=sys.stderr)
    window_order, interlace = mas_params
    i_chain, thin, n_samples, n_warmup, accept_rate, mcmc_seed, i_contd = mcmc_params
    if i_contd > 0:
        n_warmup = 1
    if 'fixed_log_Perr' in err_params.keys():
        fixed_log_Perr = err_params.pop('fixed_log_Perr')
        print('fixed_log_Perr = ', fixed_log_Perr, file=sys.stderr)

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
    
    ### call Zenbu to compute the transfer function
    if 'cubic' in model_name:
        print('test computation for cubic spectra', file=sys.stderr)
        f_model.call_Zenbu(cosmo_params_tmp, kmax)
        t1 = time.time()
        ptable_tmp = f_model.transfer_function(cosmo_params_tmp, 0.9*linear_pk[1])
        t2 = time.time()
        print(t2-t1, file=sys.stderr)
        t1 = time.time()
        ptable_tmp = f_model.transfer_function(cosmo_params_tmp, 1.1*linear_pk[1])
        t2 = time.time()
        print(t2-t1, file=sys.stderr)

    ### find indeces of independent modes
    if kmax > 1.0:
        kmax = int(kmax)
        idx_conjugate_real_kmax, idx_conjugate_imag_kmax = coord.indep_coord_stack(kmax)
        ### before applying this the model must be reduced to be kmax^3
    else:
        idx_conjugate_real, idx_conjugate_imag = coord.indep_coord_stack(ng)
        idx_kmax = coord.kmax_modes(ng, boxsize, kmax)
        idx_conjugate_real_kmax = jnp.unique(jnp.concatenate([idx_conjugate_real, idx_kmax]))
        idx_conjugate_imag_kmax = jnp.unique(jnp.concatenate([idx_conjugate_imag, idx_kmax]))
   
    print('idx_conjugate_real_kmax.shape = ', idx_conjugate_real_kmax.shape, file=sys.stderr)
    print('idx_conjugate_imag_kmax.shape = ', idx_conjugate_imag_kmax.shape, file=sys.stderr)
        
    
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
        datak = np.load(data_path) ### data = signal + noise
        print('Done.', file=sys.stderr)
    else:
        datak = data_path
    
    ### Oberved data in 1d independent modes
    if kmax > 1.0:
        ng_max = kmax
        datak_max = coord.reduce_deltak(ng_max, datak)
        datak_1d_ind = independent_modes(datak_max)
    else:
        if ng < ng_E:
            datak_E = coord.reduce_deltak(ng, datak)
        elif ng > ng_E:
            datak_E = coord.func_extend(ng, datak)
        else:
            datak_E = datak
        datak_1d_ind = independent_modes(datak_E)
       
    print('datak_1d_ind.shape = ', file=sys.stderr)
    print(datak_1d_ind.shape, file=sys.stderr)
    
    ###k2 in 1d independent modes
    k_NL = 0.1
    if kmax > 1.0:
        kvec = coord.rfftn_kvec([ng_max, ng_max, ng_max], boxsize)
    else:
        kvec = coord.rfftn_kvec([ng, ng, ng], boxsize)
    k2 = coord.rfftn_k2(kvec)
    mu2 = kvec[2]*kvec[2]/k2
    k2_1d_ind = independent_modes(k2)/(k_NL*k_NL) ###normalized by (k/k_NL)^2
    mu2_1d_ind = independent_modes(mu2) ###normalized by (k/k_NL)^2
    
    if 'Sigma2' in model_name:
        kvec_E = coord.rfftn_kvec([ng_E, ng_E, ng_E], boxsize)
        k2_E = coord.rfftn_k2(kvec_E)
        mu2_E = kvec_E[2]*kvec_E[2]/k2_E
        del kvec_E

    def model(deltak_data):
        if which_ics=='varied_ics':
            gauss_1d = numpyro.sample("gauss_1d", dist.Normal(0.0, 1.0), sample_shape=(ng3,))
            gauss_3d = coord.gauss_1d_to_3d(gauss_1d, ng)
            #gauss_1d_re, gauss_1d_im = coord.gauss_to_delta(gauss_1d, ng)
            #gauss_3d = gauss_1d_re.reshape(ng,ng,ngo2+1) + 1j*gauss_1d_im.reshape(ng,ng,ngo2+1)
        
        if 'cosmo' in which_pk:
            if 'oc' in cosmo_params.keys():
                #scaled_oc = numpyro.sample('scaled_oc', dist.Uniform(0.0, 1.0))
                omega_c = numpyro.sample('oc', dist.Uniform(0.05, 0.355))
                #omega_c = numpyro.deterministic('oc', 0.05 + (0.355 - 0.05)*scaled_oc)
            else:
                omega_c = 0.11933
            if 'hubble' in cosmo_params.keys():
                #scaled_h = numpyro.sample('scaled_hubble', dist.Uniform(0.0, 1.0))
                hubble = numpyro.sample('hubble', dist.Uniform(0.64, 0.82))
                #hubble = numpyro.deterministic('hubble', 0.64 + (0.82 - 0.64)*scaled_h)
                H0 = numpyro.deterministic('H0', hubble*100)
            else:
                hubble = 0.73
            #omega_b = numpyro.sample('ob', dist.Uniform(0.01875, 0.02625))
            omega_b = 0.02242
            #ns = numpyro.sample('ns', dist.Uniform(0.84, 1.1))
            OM = numpyro.deterministic('OM', (omega_b + omega_c)/hubble/hubble)
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
                sigma8 = numpyro.sample('sigma8', dist.Uniform(true_sigma8*0.5, true_sigma8*1.5))
                pk_lin = f_model.linear_power(cosmo_params_local)
                tmp_sigma8 = f_model.sigma8(pk_lin, type_integ='trap')
                A = numpyro.deterministic('A', sigma8/tmp_sigma8)
            elif 'A' in cosmo_params.keys():
                A = numpyro.sample('A', dist.Uniform(0.5, 1.5))
                pk_lin = f_model.linear_power(cosmo_params_local)
                sigma8 = numpyro.deterministic('sigma8', A * f_model.sigma8(pk_lin, type_integ='trap'))
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
        if 'b1' in bias_params.keys():
            true_b1 = bias_params['b1']
            #b1 = numpyro.sample('b1', dist.Uniform(0.1, 9.0))
            b1 = numpyro.sample('b1', dist.Uniform(true_b1-2, true_b1+2))
        else:
            if 'bL1' in model_name:
                b1 = 1.0
            else:
                b1 = 2.0
        
        if 'b2' in bias_params.keys():
            true_b2 = bias_params['b2']
            b2 = numpyro.sample('b2', dist.Normal(true_b2, 2.))
        else:
            b2 = -0.5
            
        if 'bG2' in bias_params.keys():
            true_bG2 = bias_params['bG2']
            bG2 = numpyro.sample('bG2', dist.Normal(true_bG2, 2.))
        else:
            bG2 = -0.5
            
        if 'bGamma3' in bias_params.keys():
            true_bGamma3 = bias_params['bGamma3']
            bGamma3 = numpyro.sample('bGamma3', dist.Normal(true_bGamma3, 2.))
        else:
            bGamma3 = -0.5
                    
        ### counter terms
        if 'cs2' in bias_params.keys():
            true_cs2 = bias_params['cs2']
            cs2 = numpyro.sample('cs2', dist.Normal(true_cs2, 20.))
        else:
            cs2 = 0.0
        if 'c1' in bias_params.keys():
            true_c1 = bias_params['c1']
            c1 = numpyro.sample('c1', dist.Normal(true_c1, 20.))
        else:
            c1 = 0.0
            
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
        
        if 'lin' in model_name or 'bL1' in model_name or 'gauss_rsd' in model_name:
            biases = [b1,]
        elif 'quad' in model_name:
            biases = [b1, b2, bG2,]
        elif 'cubic' in model_name:
            biases = [b1, b2, bG2, bGamma3,]
        else:
            biases = []
        
        if 'cs2' in model_name:
            biases += [cs2,]
        if 'c1' in model_name:
            biases += [c1, ]
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
            true_log_Perr = err_params['log_Perr']
            true_Perr = jnp.exp(true_log_Perr)
            log_Perr = numpyro.sample("log_Perr", dist.Normal(true_log_Perr, 0.5))
            Perr = jnp.exp(log_Perr)
            ratio_err = Perr/true_Perr
            Pres = numpyro.deterministic("Pres", ratio_err - 1.0)
        else:
            Perr = jnp.exp(fixed_log_Perr)
        if 'log_Perr_k2mu2' in err_params.keys():
            true_log_Perr_k2mu2 = err_params['log_Perr_k2mu2']
            log_Perr_k2mu2 = numpyro.sample("log_Perr_k2mu2", dist.Normal(true_log_Perr_k2mu2, 0.5))
            Perr_k2mu2 = jnp.exp(log_Perr_k2mu2)
            ratio_ani = numpyro.deterministic("ratio_ani", Perr_k2mu2/Perr)
            Perr += Perr*Perr_k2mu2*k2_1d_ind*mu2_1d_ind/(k_NL*k_NL)
            
        if 'Perr_k2'in err_params.keys():
            true_Perr_k2 = err_params['Perr_k2']
            Perr_k2 = numpyro.sample("Perr_k2", dist.Normal(0., 20.0))
            Perr += Perr*Perr_k2*k2_1d_ind
            
        sigma_err = jnp.sqrt(Perr/(2.*vol))

        ### for the density-dependent noise
        if 'log_Peded' in err_params.keys():
            true_log_Peded = err_params['log_Peded']
            log_Peded = numpyro.sample("log_Peded", dist.Normal(true_log_Peded, 0.5))
            Peded = numpyro.deterministic("Peded", jnp.exp(log_Peded))
            sigma_eded = jnp.sqrt(Peded/(2.*vol))
            bound = jnp.sqrt(Perr*Peded)
        #if 'Peed' in err_params.keys():
            scaled_Peed = numpyro.sample("scaled_Peed", dist.Uniform(-1, 1.))
            Peed = numpyro.deterministic("Peed", scaled_Peed*bound)
            ratio = numpyro.deterministic("ratio", Peed/Peded)
            Perr_eff = numpyro.deterministic("Perr_eff", Perr - Peed*Peed/Peded)
            sigma_err = jnp.sqrt(Perr_eff/(2.*vol))
            
        if 'b1_noise' in model_name:            
            gauss_1d_e = numpyro.sample("gauss_1d_e", dist.Normal(0.0, 1.0), sample_shape=(ng_e**3,))
            gauss_1d_e *= sigma_eded
            gauss_1d_e_re, gauss_1d_e_im = coord.gauss_to_delta(gauss_1d_e, ng_e)
            gauss_3d_e = gauss_1d_e_re.reshape(ng_e, ng_e, int(ng_e/2)+1) + 1j*gauss_1d_e_im.reshape(ng_e, ng_e, int(ng_e/2)+1)
            gauss_3d_e = gauss_3d_e.at[0,0,0].set(0.0)

        ### construct the linear field
        if which_ics=='varied_ics':
            delk = A * f_model.linear_modes(cosmo_params_local, gauss_3d)
        else:
            delk = A * f_model.linear_modes(cosmo_params_local, true_gauss_3d)

        delk_L = coord.func_extend(ng_L, delk)
        
        if 'b1_noise' in model_name:
            noisek_L = coord.func_extend(ng_L, gauss_3d_e)
            delk_L = [delk_L, noisek_L]
        
        if 'cubic' in model_name:
            fieldk_model_E = f_model.models(delk_L, biases, pk_lin)
        else:
            fieldk_model_E = f_model.models(delk_L, biases)
                                    
        if 'Sigma2_mu2' in model_name:
            fieldk_model_E *= jnp.exp(-0.5*Sigma2_mu2*k2_E*mu2_E)
        if 'Sigma2' in model_name:
            fieldk_model_E *= jnp.exp(-0.5*Sigma2*k2_E)

        if kmax > 1.0:
            fieldk_model = coord.reduce_deltak(ng_max, fieldk_model_E)
        else:
            if ng < ng_E:
                fieldk_model = coord.reduce_deltak(ng, fieldk_model_E)
            elif ng == ng_E:
                fieldk_model = fieldk_model_E
            elif ng > ng_E:
                fieldk_model = coord.func_extend(ng, fieldk_model_E)
        fieldk_model_1d_ind = independent_modes(fieldk_model)
        if 'b1_noise' in model_name:
            if kmax > 1.0:
                ed_red = coord.reduce_deltak(ng_max, noisek_L)
            else:
                ed_red = coord.reduce_deltak(ng, noisek_L)
            ed_1d_ind = independent_modes(ed_red)

        if 'b1_noise' in model_name:
            Y = numpyro.sample('Y', dist.Normal(fieldk_model_1d_ind + ratio*ed_1d_ind, sigma_err), obs=deltak_data)
        else:
            Y = numpyro.sample('Y', dist.Normal(fieldk_model_1d_ind, sigma_err), obs=deltak_data)

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
    
    if collect_ics==1:
        params += ['gauss_1d']
    elif collect_ics==2:
        params += ['gauss_1d']
        params += ['gauss_1d_e']

    min_pe_params = params.copy()
    if which_ics=='varied_ics':
        min_pe_params += ['gauss_1d']
    if 'b1_noise' in model_name:
        min_pe_params += ['gauss_1d_e']

    print('save params = ', params, file=sys.stderr)
    print('min_pe_params = ', min_pe_params, file=sys.stderr)
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
                              num_chains=1,
                              thinning=thin,
                              chain_method="sequential",
                              progress_bar=False)
    
    posterior_samples = {}
    min_pe_samples = {}
    #max_pos_samples = {}
                        
    save_base = f'{save_path}'
    print('save_base = ', save_base, file=sys.stderr)

    if i_contd > 0:
        samples_previous = np.loadtxt(f'{save_base}_{params[0]}_chain{i_chain}.dat')
        print(f'{params[0]} samples_previous.shape = ', samples_previous.shape, file=sys.stderr)
        i_contd_check = int(samples_previous.shape[0]/i_sample)
        print('i_contd = ', i_contd, file=sys.stderr)
        if i_contd != i_contd_check:
            print('i_contd != number of samples_previous * i_sample', file=sys.stderr)
            sys.exit(1)
        else:
            print('i_contd == number of samples_previous * i_sample', file=sys.stderr)

    if i_contd > 0:
        rng_key = jax.random.PRNGKey(0)
        mcmc.warmup(rng_key, deltak_data=datak_1d_ind, extra_fields=('potential_energy',))
        print('Restarting samping...', file=sys.stderr)
        with open(f'{save_base}_{i_chain}_{i_contd}_last_state.pkl', 'rb') as f:
            last_state = pickle.load(f)
        mcmc.post_warmup_state = last_state
        mcmc._last_state = last_state
        print('LOADED warmup state = ', file=sys.stderr)
        print(mcmc.post_warmup_state, file=sys.stderr)
        print('LOADED last state = ', file=sys.stderr)
        print(mcmc.last_state, file=sys.stderr)
        min_pe = np.loadtxt(f'{save_base}_min_pe_chain{i_chain}.dat')
        for param in min_pe_params:
            min_pe_samples[param] = np.loadtxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat')
    #elif i_chain > 0 :
    #    rng_key = jax.random.PRNGKey(0)
    #    mcmc.warmup(rng_key, deltak_data=datak_1d_ind, extra_fields=('potential_energy',))
    #    print(f'Loading post warmup state from {save_base}_0_{mcmc_seed}_warmup_state.pkl ...', file=sys.stderr)
    #    with open(f'{save_base}_0_{mcmc_seed}_warmup_state.pkl', 'rb') as f:
    #        warmup_state = pickle.load(f)
    #    mcmc.post_warmup_state = warmup_state
    #    mcmc._last_state = warmup_state
    #    print('LOADED warmup state = ', file=sys.stderr)
    #    print(mcmc.post_warmup_state, file=sys.stderr)
    #    print('LOADED last state = ', file=sys.stderr)
    #    print(mcmc.last_state, file=sys.stderr)
    elif n_warmup == 1:
        rng_key = jax.random.PRNGKey(0)
        mcmc.warmup(rng_key, deltak_data=datak_1d_ind, extra_fields=('potential_energy',))
        print(f'Loading post warmup state from {save_base}_0_{mcmc_seed}_warmup_state.pkl ...', file=sys.stderr)
        with open(f'{save_base}_0_{mcmc_seed}_warmup_state.pkl', 'rb') as f:
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
        mcmc.warmup(rng_key, deltak_data=datak_1d_ind, extra_fields=('potential_energy',), collect_warmup=False)
        inv_mass_matrix = mcmc.post_warmup_state.adapt_state.inverse_mass_matrix
    
        i_warmup = 1
        print(inv_mass_matrix, file=sys.stderr)
        print('i_warmup = ', i_warmup, file=sys.stderr)
       
        criteria = 1.0
        
        if which_ics=='varied_ics':
            while_test = (jnp.abs(inv_mass_matrix[dense_mass[0]][0,0]) > criteria) or (jnp.abs(inv_mass_matrix[dense_mass[0]][1,1]) > criteria)
        else:
            while_test = (jnp.abs(inv_mass_matrix[dense_mass[0]][0,0]) > criteria)
            
        print('while = ', while_test, file=sys.stderr)
        
        with open(f'{save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl', 'wb') as f:
            pickle.dump(mcmc.post_warmup_state, f)
        
        while( while_test ):
            rng_key = jax.random.PRNGKey(mcmc_seed+i_warmup)
            print('rng_seed = ',mcmc_seed+i_warmup, file=sys.stderr)
            print('rng key = ', rng_key, file=sys.stderr)
            mcmc.warmup(rng_key, deltak_data=datak_1d_ind, extra_fields=('potential_energy',))
            inv_mass_matrix = mcmc.post_warmup_state.adapt_state.inverse_mass_matrix
            i_warmup += 1
            print(inv_mass_matrix, file=sys.stderr)
            
            while_test = (jnp.abs(inv_mass_matrix[dense_mass[0]][0,0]) > criteria) or (jnp.abs(inv_mass_matrix[dense_mass[0]][1,1]) > criteria)
            print('while = ', while_test, file=sys.stderr)
            
            print('i_warmup = ', i_warmup, file=sys.stderr)
        
            with open(f'{save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl', 'wb') as f:
                pickle.dump(mcmc.post_warmup_state, f)

    for i in range(i_iter):
        print(f'running batch {i} ...', file=sys.stderr)
        rng_key = jax.random.PRNGKey(mcmc_seed+12*i_chain+123*i+1234*i_contd)
        mcmc.run(rng_key, deltak_data=datak_1d_ind, extra_fields=('potential_energy',))
        samples = mcmc.get_samples()
        pes = mcmc.get_extra_fields()['potential_energy']
        min_pe_idx = jnp.argmin(pes)
        if i == 0:
            if(i_contd==0):
                min_pe = pes[min_pe_idx]
                print('Min of the potential energy = ', min_pe, file=sys.stderr)
                for param in min_pe_params:
                    min_pe_samples[param] = samples[param][min_pe_idx]
                    print(f'min_pe_samples[{param}] = ', min_pe_samples[param], file=sys.stderr)
            if which_ics=='varied_ics':
                mean_gauss_1d = jnp.mean(samples['gauss_1d'], axis=0)
            if 'b1_noise' in model_name:
                mean_gauss_1d_e = jnp.mean(samples['gauss_1d_e'], axis=0)
            for param in params:
                posterior_samples[param] = jax.device_put(samples[param].astype(np.float32), cpus[0])
            posterior_samples['potential_energy'] = jax.device_put(pes.astype(np.float32), cpus[0])
        else:
            if min_pe > pes[min_pe_idx]:
                min_pe = pes[min_pe_idx]
                print('Min of the potential energy = ', min_pe, file=sys.stderr)
                for param in min_pe_params:
                    min_pe_samples[param] = samples[param][min_pe_idx]
                    print(f'min_pe_samples[{param}] = ', min_pe_samples[param], file=sys.stderr)
            if which_ics=='varied_ics':
                tmp_mean = jnp.mean(samples['gauss_1d'], axis=0)
                mean_gauss_1d = ( i*mean_gauss_1d + tmp_mean )/(i+1)
            if 'b1_noise' in model_name:
                tmp_mean_e = jnp.mean(samples['gauss_1d_e'], axis=0)
                mean_gauss_1d_e = ( i*mean_gauss_1d_e + tmp_mean_e )/(i+1)
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

    if 'b1_noise' in model_name:
        if(i_contd>0):
            mean_gauss_1d_e_previous = np.loadtxt(f'{save_base}_gauss_1d_e_mean_chain{i_chain}.dat')
            mean_gauss_1d_e = ( n_samples*mean_gauss_1d_e + i_contd*int(i_sample/thin)*mean_gauss_1d_e_previous )/(n_samples + i_contd*int(i_sample/thin))
        np.savetxt(f'{save_base}_gauss_1d_e_mean_chain{i_chain}.dat', mean_gauss_1d_e)

    np.savetxt(f'{save_base}_min_pe_chain{i_chain}.dat', np.array([min_pe]))
    for param in min_pe_params:
        print(f'min_pe_sample[{param}]', min_pe_samples[param].shape, file=sys.stderr)
        if param=='gauss_1d':
            np.savetxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat', min_pe_samples[param])
        elif param=='gauss_1d_e':
            np.savetxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat', min_pe_samples[param])
        else:
            np.savetxt(f'{save_base}_{param}_min_pe_chain{i_chain}.dat', np.array([min_pe_samples[param]]))

    if i_chain > 0:
        os.remove(f'{save_base}_{i_chain}_{mcmc_seed}_warmup_state.pkl')
