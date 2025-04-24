#!/usr/bin/env python3
import os
import sys
import copy
import pickle
import logging
import numpy as np
from typing import Optional

import jax
import jax.numpy as jnp
from jax import random, jit, lax, pmap
from jax.tree_util import tree_map
import jax.scipy as jsp
from functools import partial

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from functools import partial

# Custom modules
import field_level.coord as coord
import field_level.forward_model as forward_model
import field_level.util as utiil
import field_level.cosmo_util as cosmo_util
import field_level.power_util as power_util

# Logging configuration
logging.basicConfig(stream=sys.stderr,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# JAX configuration
if jax.config.read('jax_enable_x64'):
    numpyro.enable_x64()
    logger.info('NumPyro x64 mode is enabled because JAX is in x64 mode.')

logger.info(f'The inference is running on {jax.default_backend()}')
cpus = jax.devices('cpu')
gpus = jax.devices('gpu')

# Constants (can be parameterized as needed)
DEFAULT_I_SAMPLE = 100
MAX_WARMUP_ITERS = 10  # For preventing infinite warmup loop

OMEGA_B_TRUE = 0.02242
OMEGA_C_TRUE = 0.11933
HUBBLE_TRUE = 0.73
NS_TRUE = 0.9665

# -----------------------------
# Helper Functions
# -----------------------------
def sample_uniform_deterministic(param_name, scaled_name, bounds):
    """
    Sample a value from a Uniform distribution and record it as a deterministic value.
    """
    min_val, max_val = bounds
    scaled = numpyro.sample(scaled_name, dist.Uniform(0.0, 1.0))
    value = numpyro.deterministic(param_name, min_val + (max_val - min_val) * scaled)
    return value

def check_dense_mass_matrix(inv_mass_matrix, dense_mass, criteria=0.9):
    """
    Check each parameter (except 'c0', 'c2', 'c4') in the mass matrix block corresponding to dense_mass.
    Returns an index mapping and a boolean indicating if any diagonal value exceeds the criteria.
    """
    block = inv_mass_matrix[dense_mass]
    check_fail = False
    index_map = {}
    for i, param in enumerate(dense_mass):
        if param in ['c0', 'c2', 'c4']:
            continue
        index_map[param] = i
        diag_val = jnp.abs(block[:, i, i])
        if (diag_val > criteria).any():
            logger.info(f'Parameter {param} (index {i}) has diag value {diag_val} exceeding criteria {criteria}')
            check_fail = True
        else:
            logger.info(f'Parameter {param} (index {i}) OK: diag value {diag_val}')
    return index_map, check_fail

def split_hmc_state_by_chain(hmc_state, num_chains):
    """
    Split a batched HMCState by each chain and return as a list.
    """
    single_chain_states = []
    if num_chains == 1:
        single_chain_states.append(hmc_state)
    else:
        for c in range(num_chains):
            def select_chain(x):
                if hasattr(x, 'shape') and x.shape[0] == num_chains:
                    return x[c]
                else:
                    return x
            single_chain_state = tree_map(select_chain, hmc_state)
            single_chain_states.append(single_chain_state)
    return single_chain_states

def merge_hmc_states(multi_state, single_state, i_chain):
    """
    Merge a single-chain HMCState (single_state) into multi_state at the specified chain index.
    """
    def _update(multi_val, single_val):
        if hasattr(multi_val, 'shape') and len(multi_val.shape) >= 1 and multi_val.shape[0] > i_chain:
            new_arr = jnp.array(multi_val)
            new_arr = new_arr.at[i_chain].set(jnp.array(single_val))
            return new_arr
        else:
            return multi_val
    new_multi_state = tree_map(_update, multi_state, single_state)
    return new_multi_state

def load_data(data_path):
    """
    Load data from file if data_path is a string.
    """
    if isinstance(data_path, str):
        logger.info(f'Loading data from {data_path}')
        try:
            data = np.load(data_path)
        except Exception as e:
            logger.error(f'Failed to load data from {data_path}: {e}')
            raise
        logger.info('Data loaded successfully.')
    else:
        data = data_path
    return data

def compute_pk(name: str,
              fieldk1: jnp.ndarray,
              fieldk2: Optional[jnp.ndarray],
              measure_pk: power_util.Measure_Pk):
    """
    # Compute auto- or cross- power spectrum and record it as numpyro.deterministic.
    #
    # name     : the key under which to store the [2, nbin] array
    # fieldk1  : complex FFT field for auto (or first field in cross)
    # fieldk2  : iff None → auto, else → cross against this field
    # measure_pk: Measure_Pk instance
    """
    if fieldk2 is None:
        k, pk, _ = measure_pk.pk_auto(fieldk1)
    else:
        k, pk, _ = measure_pk.pk_cross(fieldk1, fieldk2)
    # stack k and pk so the user can inspect both
    out = jnp.stack([k.real, pk.real], axis=0)
    numpyro.deterministic(name, out)

# -----------------------------
# Main Inference Function
# -----------------------------
def field_inference(boxsize, redshift, which_pk,
                    data_path, save_path,
                    ics_params, model_name, ng_params, mas_params, which_space,
                    cosmo_params, bias_params, err_params, kmax, 
                    dense_mass, mcmc_params, 
                    pk_params=None, true_gauss_3d=None,
                    **kwargs):
    r"""
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
        Each value should be a tuple of (min, max) or (mean, std) for prior.

    err_params : dict
        The error (in the likelihood) parameters to sample (or not). The keys should be
            'log_Perr' : The logarithm of the (white) noise power spectrum
            'fixed_log_Perr' : The logarithm of the (white) noise power spectrum, and will not sample it.
        The value of 'log_Perr' should (mean, std) for prior.
    
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
    logger.info(f'Initial IC settings: {which_ics}')
    window_order, interlace = mas_params
    i_chain, n_chains, thin, n_samples, n_warmup, accept_rate, mcmc_seed, i_contd = mcmc_params
    numpyro.set_host_device_count(n_chains)
    if i_contd > 0:
        n_warmup = 1

    ### flags for pk_ic computations
    if isinstance(pk_params, (list, tuple)) and len(pk_params)==3:
        pk_nbin, pk_kmin, pk_kmax = pk_params
        do_pk = pk_nbin is not None
    else:
        do_pk = False

    # Handle fixed error parameters
    if 'fixed_log_Perr' in err_params:
        fixed_log_Perr = err_params.pop('fixed_log_Perr')
        logger.info(f'fixed_log_Perr = {fixed_log_Perr}')
    if 'fixed_log_Peed' in err_params:
        fixed_log_Peed = err_params.pop('fixed_log_Peed')
        logger.info(f'fixed_log_Peed = {fixed_log_Peed}')
    if 'fixed_Peded' in err_params:
        fixed_Peded = err_params.pop('fixed_Peded')
        logger.info(f'fixed_Peded = {fixed_Peded}')

    vol = boxsize**3

    # Load fixed initial conditions if applicable
    if which_ics != 'varied_ics':
        logger.info(f'Loading the true initial conditions from {which_ics}')
        try:
            true_gauss_3d = np.load(which_ics)
        except Exception as e:
            logger.error(f'Failed to load true ICs from {which_ics}: {e}')
            raise
        logger.info('Initial conditions loaded.')
    else:
        if true_gauss_3d is not None:
            true_gauss_3d = jnp.array(load_data(true_gauss_3d))

    # Determine window function string
    if window_order == 1:
        w_str = 'ngp'
    elif window_order == 2:
        w_str = 'cic'
    elif window_order == 3:
        w_str = 'tsc'
    logger.info('%s interlacing %s', w_str, 'on' if interlace==1 else 'off')

    # Unpack ng_params
    if len(ng_params) == 3:
        ng, ng_L, ng_E = map(int, ng_params)
    elif len(ng_params) == 5:
        ng, ng_L, ng_E, ng_cut, ng_e = ng_params
    logger.info("ng = %d, ng_L = %d, ng_E = %d, kmax = %s", ng, ng_L, ng_E, kmax)

    ng3 = int(ng**3)

    if do_pk:
        kbin_1d = jnp.linspace(pk_kmin, pk_kmax, pk_nbin+1)
        measure_pk = power_util.Measure_Pk(boxsize, ng, kbin_1d)
        dummy = jnp.ones((ng, ng, ng // 2 + 1)) + 0.*1j
        _, _, _ = measure_pk.pk_auto(dummy)
        _, _, _ = measure_pk.pk_cross(dummy, dummy)

    # Create the forward model
    if 'lpt' in model_name:
        f_model = forward_model.Forward_Model(model_name, which_pk, ng_params, boxsize, which_space, mas_params=mas_params)
    else:
        f_model = forward_model.Forward_Model(model_name, which_pk, ng_params, boxsize, which_space)

    # Prepare k-vectors
    f_model.kvecs(kmax)

    # Compute indices for independent modes (for k_space)
    if which_space == 'k_space':
        if kmax > 1.0:
            kmax = int(kmax)
            ng_max = kmax
            idx_conjugate_real_kmax, idx_conjugate_imag_kmax = coord.indep_coord_stack(ng_max)
        else:
            idx_conjugate_real, idx_conjugate_imag = coord.indep_coord_stack(ng)
            idx_kmax = coord.above_kmax_modes(ng, boxsize, kmax)
            idx_conjugate_real_kmax = np.unique(np.concatenate([idx_conjugate_real, idx_kmax]))
            idx_conjugate_imag_kmax = np.unique(np.concatenate([idx_conjugate_imag, idx_kmax]))
        logger.info('idx_conjugate_real_kmax.shape = %s', idx_conjugate_real_kmax.shape)
        logger.info('idx_conjugate_imag_kmax.shape = %s', idx_conjugate_imag_kmax.shape)
    elif which_space == 'r_space':
        kmax = int(kmax)
        ng_max = kmax
        ng3_max = ng_max**3

    @jit
    def independent_modes(fieldk):
        fieldk_1d = fieldk.ravel()
        fieldk_real_1d_ind = jnp.delete(fieldk_1d.real, idx_conjugate_real_kmax)
        fieldk_imag_1d_ind = jnp.delete(fieldk_1d.imag, idx_conjugate_imag_kmax)
        fieldk_1d_ind = jnp.hstack([fieldk_real_1d_ind, fieldk_imag_1d_ind])
        return fieldk_1d_ind

    # Load the data
    data = load_data(data_path)

    # Preprocess data to 1D (depending on space)
    if which_space == 'k_space':
        data[0,0,0] = 0.0  # Remove DC mode
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
            logger.info('Data is in Fourier space.')
            data[0,0,0] = 0.0
            data_ = coord.func_reduce(ng_max, data)
            datar = jnp.fft.irfftn(data_) * (ng_max**3)
            data_1d_ind = datar.reshape(ng_max**3)
        else:
            data_1d_ind = data.reshape(ng_max**3)
            data_1d_ind -= data_1d_ind.mean()
        logger.info('datar_mean = %s', data_1d_ind.mean())

    logger.info('data_1d_ind.shape = %s', data_1d_ind.shape)

    # Calculate k2 and mu2 for k_space
    k_NL = 0.1
    if which_space == 'k_space':
        if kmax > 1.0:
            kvec = coord.rfftn_kvec([ng_max]*3, boxsize)
        else:
            kvec = coord.rfftn_kvec([ng]*3, boxsize)
        k2 = coord.rfftn_k2(kvec)
        k2_1d_ind = independent_modes(k2) / (k_NL**2)
        k2 = k2.at[0,0,0].set(1.0)
        mu2 = kvec[2]**2 / k2
        mu2_1d_ind = independent_modes(mu2)
    if 'Sigma2' in model_name:
        kvec_E = coord.rfftn_kvec([ng_E]*3, boxsize)
        k2_E = coord.rfftn_k2(kvec_E)
        mu2_E = kvec_E[2]**2 / k2_E
        del kvec_E

    # -----------------------------
    # Model definition
    # -----------------------------
    def model(obs_data):
        if which_ics == 'varied_ics':
            gauss_1d = numpyro.sample("gauss_1d", dist.Normal(0.0, 1.0), sample_shape=(ng3,))
            gauss_3d = coord.gauss_1d_to_3d(gauss_1d, ng)
        # The sampling for cosmo_params, bias_params, err_params
        if 'cosmo' in which_pk:
            if 'oc' in cosmo_params:
                omega_c = sample_uniform_deterministic('oc', 'scaled_oc', cosmo_params['oc'])
            else:
                omega_c = 0.11933
            if 'hubble' in cosmo_params:
                hubble = sample_uniform_deterministic('hubble', 'scaled_hubble', cosmo_params['hubble'])
                H0 = numpyro.deterministic('H0', hubble * 100)
            else:
                hubble = 0.73
            if 'ob' in cosmo_params:
                omega_b = sample_uniform_deterministic('ob', 'scaled_ob', cosmo_params['ob'])
            else:
                omega_b = 0.02242
            OM = numpyro.deterministic('OM', (omega_b + omega_c) / hubble**2)
            if 'ns' in cosmo_params:
                ns = sample_uniform_deterministic('ns', 'scaled_ns', cosmo_params['ns'])
            else:
                ns = 0.9665
            ln1010As = 3.047
            cosmo_params_local = jnp.array([omega_b, omega_c, hubble, ns, ln1010As, 0.0])
            if 'sigma8' in cosmo_params:
                min_sigma8, max_sigma8 = cosmo_params['sigma8']
                scaled_sigma8 = numpyro.sample('scaled_sigma8', dist.Uniform(0.0, 1.0))
                sigma8 = numpyro.deterministic('sigma8', min_sigma8 + (max_sigma8 - min_sigma8) * scaled_sigma8)
                pk_lin = f_model.linear_power(cosmo_params_local)
                tmp_sigma8 = f_model.sigmaR(pk_lin)
                A = numpyro.deterministic('A', sigma8 / tmp_sigma8)
            elif 'A' in cosmo_params:
                A = numpyro.sample('A', dist.Uniform(*cosmo_params['A']))
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
            if 'A' in cosmo_params:
                A = numpyro.sample('A', dist.Uniform(*cosmo_params['A']))
            else:
                A = 1.0

        A2 = A * A
        A3 = A * A * A

        # Bias parameters sampling
        if 'Ab1' in bias_params:
            Ab1 = numpyro.sample('Ab1', dist.Uniform(*bias_params['Ab1']))
            b1 = numpyro.deterministic('b1', Ab1 / A)
        elif 'b1' in bias_params:
            b1 = numpyro.sample('b1', dist.Uniform(*bias_params['b1']))
            Ab1 = numpyro.deterministic('Ab1', A * b1)
        else:
            b1 = 1.0
        if 'A2b2' in bias_params:
            A2b2 = numpyro.sample('A2b2', dist.Normal(*bias_params['A2b2']))
            b2 = numpyro.deterministic('b2', A2b2 / A2)
        elif 'b2' in bias_params:
            b2 = numpyro.sample('b2', dist.Normal(*bias_params['b2']))
            A2b2 = numpyro.deterministic('A2b2', A2 * b2)
        else:
            b2 = 0.0
        if 'A2bG2' in bias_params:
            A2bG2 = numpyro.sample('A2bG2', dist.Normal(*bias_params['A2bG2']))
            bG2 = numpyro.deterministic('bG2', A2bG2 / A2)
        elif 'bG2' in bias_params:
            bG2 = numpyro.sample('bG2', dist.Normal(*bias_params['bG2']))
            A2bG2 = numpyro.deterministic('A2bG2', A2 * bG2)
        else:
            bG2 = 0.0
        if 'A3bGamma3' in bias_params:
            A3bGamma3 = numpyro.sample('A3bGamma3', dist.Normal(*bias_params['A3bGamma3']))
            bGamma3 = numpyro.deterministic('bGamma3', A3bGamma3 / A3)
        elif 'bGamma3' in bias_params:
            bGamma3 = numpyro.sample('bGamma3', dist.Normal(*bias_params['bGamma3']))
            A3bGamma3 = numpyro.deterministic('A3bGamma3', A3 * bGamma3)
        else:
            bGamma3 = 0.0
        if 'A3b3' in bias_params:
            A3b3 = numpyro.sample('A3b3', dist.Normal(*bias_params['A3b3']))
            b3 = numpyro.deterministic('b3', A3b3 / A3)
        elif 'b3' in bias_params:
            b3 = numpyro.sample('b3', dist.Normal(*bias_params['b3']))
            A3b3 = numpyro.deterministic('A3b3', A3 * b3)
        else:
            b3 = 0.0
        if 'A3bG2d' in bias_params:
            A3bG2d = numpyro.sample('A3bG2d', dist.Normal(*bias_params['A3bG2d']))
            bG2d = numpyro.deterministic('bG2d', A3bG2d / A3)
        elif 'bG2d' in bias_params:
            bG2d = numpyro.sample('bG2d', dist.Normal(*bias_params['bG2d']))
            bG2d = numpyro.deterministic('bG2d', A3 * bG2d)
        else:
            bG2d = 0.0
        if 'A3bG3' in bias_params:
            A3bG3 = numpyro.sample('A3bG3', dist.Normal(*bias_params['A3bG3']))
            bG3 = numpyro.deterministic('bG3', A3bG3 / A3)
        elif 'bG3' in bias_params:
            bG3 = numpyro.sample('bG3', dist.Normal(*bias_params['bG3']))
            bG3 = numpyro.deterministic('bG3', A3 * bG3)
        else:
            bG3 = 0.0

        # Counter terms
        if 'c0' in bias_params:
            c0 = numpyro.sample('c0', dist.Normal(*bias_params['c0']))
        else:
            c0 = -2.705611
        if 'c2' in bias_params:
            c2 = numpyro.sample('c2', dist.Normal(*bias_params['c2']))
        else:
            c2 = -11.00391
        if 'c4' in bias_params:
            c4 = numpyro.sample('c4', dist.Normal(*bias_params['c4']))
        else:
            c4 = 5.498273
        if 'Sigma2' in bias_params:
            Sigma2 = numpyro.sample('Sigma2', dist.Normal(*bias_params['Sigma2']))
        else:
            Sigma2 = 0.0
        if 'Sigma2_mu2' in bias_params:
            Sigma2_mu2 = numpyro.sample('Sigma2_mu2', dist.Normal(*bias_params['Sigma2_mu2']))
        else:
            Sigma2_mu2 = 0.0

        if 'lin' in model_name:
            biases = [b1,]
        elif 'quad' in model_name:
            biases = [b1, b2, bG2,]
        elif 'cubic' in model_name:
            biases = [b1, b2, bG2, b3, bG2d, bG3, bGamma3,]
        else:
            biases = []
        if 'c0' in model_name:
            biases += [c0,]
        if 'c2' in model_name:
            biases += [c2,]
        if 'c4' in model_name:
            biases += [c4,]
        if 'Sigma2' in model_name:
            biases += [Sigma2,]
        if 'Sigma2_mu2' in model_name:
            biases += [Sigma2_mu2,]
        if 'rsd' in model_name:
            growth_f = numpyro.deterministic('growth_f', cosmo_util.growth_f_fitting(redshift, OM))
            biases += [growth_f,]

        # Error model
        if 'log_Perr' in err_params:
            if err_params['log_Perr'][0] > 15.:
                mean_log_Perr = jnp.log((vol/err_params['log_Perr'][0])**3)
            else:
                mean_log_Perr = err_params['log_Perr'][0]
            std_log_Perr = err_params['log_Perr'][1]
            log_Perr = numpyro.sample("log_Perr", dist.Normal(mean_log_Perr, std_log_Perr))
            Perr = jnp.exp(log_Perr)
        else:
            if fixed_log_Perr > 15.:
                Perr = (vol/fixed_log_Perr)**3
            else:
                Perr = jnp.exp(fixed_log_Perr)
        if which_space == 'k_space':
            sigma_err = jnp.sqrt(Perr / (2. * vol))
        elif which_space == 'r_space':
            sigma2_err = Perr * ng3 / vol

        if 'log_Peded' in err_params:
            log_Peded = numpyro.sample("log_Peded", dist.Normal(*err_params['log_Peded']))
            Peded = numpyro.deterministic("Peded", jnp.exp(log_Peded))
            bound = jnp.sqrt(Perr * Peded)
            if which_space == 'k_space':
                sigma_eded = jnp.sqrt(Peded / (2. * vol))
            elif which_space == 'r_space':
                sigma2_eded = Peded * ng3 / vol
        elif 'fixed_log_Peded' in err_params:
            fixed_log_Peded = err_params['fixed_log_Peded']
            Peded = jnp.exp(fixed_log_Peded)
            if which_space == 'k_space':
                sigma_eded = jnp.sqrt(Peded / (2. * vol))
            elif which_space == 'r_space':
                sigma2_eded = Peded * ng3 / vol
        if 'Peed' in err_params:
            scaled_Peed = numpyro.sample("scaled_Peed", dist.Uniform(-1, 1.))
            Peed = numpyro.deterministic("Peed", scaled_Peed * bound)
            ratio = numpyro.deterministic("ratio", Peed / Peded)
            if which_space == 'k_space':
                Perr_eff = numpyro.deterministic("Perr_eff", Perr - Peed * Peed / Peded)
                sigma_err = jnp.sqrt(Perr_eff / (2. * vol))
            elif which_space == 'r_space':
                sigma2_eed = Peed * ng3 / vol
        elif 'fixed_Peed' in err_params:
            Peed = jnp.exp(fixed_log_Peed)
            if which_space == 'k_space':
                Perr_eff = numpyro.deterministic("Perr_eff", Perr - Peed * Peed / Peded)
                sigma_err = jnp.sqrt(Perr_eff / (2. * vol))
            elif which_space == 'r_space':
                sigma2_eed = Peed * ng3 / vol

        if which_ics == 'varied_ics':
            delk = A * f_model.linear_modes(cosmo_params_local, gauss_3d)
            if do_pk:
                g3 = gauss_3d / jnp.sqrt(2.*vol)
                compute_pk('pk_ic_gauss', g3, None, measure_pk)
                compute_pk('pk_ic_cosmo', delk, None, measure_pk)
                if true_gauss_3d is not None:
                    true_g3 = true_gauss_3d / jnp.sqrt(2.*vol)
                    res_g3 = true_g3 - g3
                    compute_pk('pk_ic_gauss_res', res_g3, None, measure_pk)
                    compute_pk('pk_ic_gauss_cross', g3, true_g3, measure_pk)
                    cosmo_params_true = jnp.array([OMEGA_B_TRUE, OMEGA_C_TRUE, HUBBLE_TRUE, NS_TRUE, ln1010As, redshift])
                    true_delk = f_model.linear_modes(cosmo_params_true, true_gauss_3d)
                    res_delk = delk - true_delk
                    compute_pk('pk_ic_cosmo_res', res_delk, None, measure_pk)
                    compute_pk('pk_ic_cosmo_cross', delk, true_delk, measure_pk)
        else:
            delk = A * f_model.linear_modes(cosmo_params_local, true_gauss_3d)
        delk_L = coord.func_extend(ng_L, delk)
                
        if which_space == 'k_space':
            fieldk_model_E = f_model.compute_model(delk_L, biases)
            if 'Sigma2_mu2' in model_name:
                fieldk_model_E = fieldk_model_E * jnp.exp(-0.5 * Sigma2_mu2 * k2_E * mu2_E)
            if 'Sigma2' in model_name:
                fieldk_model_E = fieldk_model_E * jnp.exp(-0.5 * Sigma2 * k2_E)
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
            fieldr_model, delr_max, d2r_max = f_model.compute_model(delk_L, biases)
            field_model_1d_ind = fieldr_model.reshape(ng3)
            if 'log_Peded' in err_params or 'fixed_log_Peded' in err_params:
                d2r_1d_ind = d2r_max.reshape(ng3)
            if 'Peed' in err_params or 'fixed_Peed' in err_params:
                delr_1d_ind = delr_max.reshape(ng3)
            if ('Peded' in err_params or 'fixed_Peded' in err_params or 
                'Peed' in err_params or 'fixed_Peed' in err_params):
                sigma2_err = sigma2_err + 2.0 * sigma2_eed * delr_1d_ind + sigma2_eded * d2r_1d_ind
            sigma_err = jnp.sqrt(sigma2_err)
        data = numpyro.sample('data', dist.Normal(field_model_1d_ind, sigma_err), obs=obs_data)

    # -----------------------------
    # Setup before MCMC execution
    # -----------------------------
    logger.info('dense_mass = %s', dense_mass)
    n_total = thin * n_samples
    i_sample = DEFAULT_I_SAMPLE
    i_iter = int(n_total / i_sample)

    kernel = NUTS(model=model,
                  target_accept_prob=accept_rate,
                  adapt_step_size=True,
                  adapt_mass_matrix=True,
                  dense_mass=dense_mass,
                  max_tree_depth=(9, 9),
                  forward_mode_differentiation=False,
                  init_strategy=numpyro.infer.init_to_sample)

    chain_method = 'sequential' if n_chains == 1 else 'parallel'

    # --- Initial MCMC warmup (num_samples=1) ---
    mcmc = MCMC(kernel, num_samples=1, num_warmup=1,
                num_chains=n_chains, thinning=thin,
                chain_method=chain_method, progress_bar=False)
    # Create an RNG key from mcmc_seed (do not repeatedly use key 0)
    rng_key = jax.random.PRNGKey(0)
    mcmc.warmup(rng_key, obs_data=data_1d_ind, extra_fields=('potential_energy',))
    params = list(mcmc.get_samples().keys())
    params.append('potential_energy')
    map_params = params.copy()
    if collect_ics == 0:
        params.remove('gauss_1d')
    logger.info('params = %s', params)
    logger.info('MAP_params = %s', map_params)
    del mcmc

    # --- Main MCMC execution (i_iter batches) ---
    mcmc = MCMC(kernel, num_samples=i_sample, num_warmup=n_warmup,
                num_chains=n_chains, thinning=thin,
                chain_method=chain_method, progress_bar=False)
    
    samples_prev = {}
    map_samples = {}

    mean_gauss_1d_prev = {}

    save_base = f'{save_path}'
    logger.info('save_base = %s', save_base)

    # Load previous samples if restarting (i_contd > 0)
    if i_contd > 0:
        for c in range(n_chains):
            i_chain_ = i_chain + c
            chain_key = f'chain_{i_chain_}'
            
            chain_file = f'{save_base}_samples_chain{i_chain_}.npz'
            if os.path.exists(chain_file):
                loaded = np.load(chain_file, allow_pickle=True)
                samples_prev[chain_key] = loaded['samples'].item()
                map_samples[chain_key] = loaded['MAP_samples'].item()
                if which_ics == 'varied_ics':
                    mean_gauss_1d_prev[chain_key] = loaded['mean_gauss_1d']
                logger.info('Loaded previous samples from %s', chain_file)
            else:
                logger.error('Previous samples file %s not found.', chain_file)
                sys.exit(1)
            ### check the iteration count
            sample_shape = samples_prev[chain_key][params[0]].shape  # shape = (total_samples_prev,)
            i_contd_check = int(sample_shape[0] / i_sample)
            logger.info(f'{params[0]} samples_previous.shape = {sample_shape}')
            logger.info(f'i_contd = {i_contd}')
            if i_contd != i_contd_check:
                logger.error('i_contd != number of iterations in previous samples')
                sys.exit(1)
            else:
                logger.info('i_contd matches previous iterations count.')
        
        rng_key = jax.random.PRNGKey(1)  # use seed, not 0
        mcmc.warmup(rng_key, obs_data=data_1d_ind, extra_fields=('potential_energy',))
        logger.info('Restarting sampling...')
        if n_chains == 1:
            with open(f'{save_base}_{i_chain}_{i_contd}_last_state.pkl', 'rb') as f:
            #with open(f'{save_base}_0_110_last_state.pkl', 'rb') as f:
                last_state = pickle.load(f)
            mcmc.post_warmup_state = last_state
        else:
            for c in range(n_chains):
                i_chain_ = i_chain + c
                with open(f'{save_base}_{i_chain_}_{i_contd}_last_state.pkl', 'rb') as f:
                    last_state = pickle.load(f)
                last_state = tree_map(jnp.array, last_state)
                mcmc.post_warmup_state = merge_hmc_states(mcmc.post_warmup_state, last_state, c)
        logger.info('LOADED warmup state:')
        logger.info(mcmc.post_warmup_state)

    elif n_warmup <= 1:
        rng_key = jax.random.PRNGKey(2)
        mcmc.warmup(rng_key, obs_data=data_1d_ind, extra_fields=('potential_energy',))
        if n_chains == 1:
            with open(f'{save_base}_{i_chain}_{mcmc_seed_}_warmup_state.pkl', 'rb') as f:
            #with open(f'{save_base}_{i_chain}_80_last_state.pkl', 'rb') as f:
            #with open(f'{save_base}_3_10_last_state.pkl', 'rb') as f:
                last_state = pickle.load(f)
            mcmc.post_warmup_state = last_state
        else:
            for c in range(n_chains):
                i_chain_ = i_chain + c
                mcmc_seed_ = mcmc_seed + 12345 * i_chain_
                with open(f'{save_base}_{i_chain_}_{mcmc_seed_}_warmup_state.pkl', 'rb') as f:
                #with open(f'{save_base}_{i_chain_}_50_last_state.pkl', 'rb') as f:
                #with open(f'{save_base}_0_50_last_state.pkl', 'rb') as f:
                    warmup_state = pickle.load(f)
                warmup_state = tree_map(jnp.array, warmup_state)
                mcmc.post_warmup_state = merge_hmc_states(mcmc.post_warmup_state, warmup_state, c)
            logger.info('LOADED warmup state:')
            logger.info(mcmc.post_warmup_state)
    else:
        mcmc_seed += 12345 * i_chain
        rng_key = jax.random.PRNGKey(mcmc_seed)
        logger.info(f'rng_seed = {mcmc_seed}')
        mcmc.warmup(rng_key, obs_data=data_1d_ind, extra_fields=('potential_energy',), collect_warmup=False)
        logger.info('post_warmup_state:')
        logger.info(mcmc.post_warmup_state)
        inv_mass_matrix = copy.deepcopy(mcmc.post_warmup_state.adapt_state.inverse_mass_matrix)
        if n_chains == 1:
            inv_mass_matrix[dense_mass[0]] = inv_mass_matrix[dense_mass[0]][None, :, :]
        i_warmup = 1
        criteria = 0.9
        logger.info(f'i_warmup = {i_warmup}')
        logger.info(f'inv_mass_matrix[{dense_mass[0]}].shape = {inv_mass_matrix[dense_mass[0]].shape}')
        logger.info(f'inv_mass_matrix = {inv_mass_matrix}')
        index_map, check_fail = check_dense_mass_matrix(inv_mass_matrix, dense_mass[0], criteria)
        logger.info(f'Initial dense mass matrix check: check_fail={check_fail}')
        post_warmup_state = tree_map(jnp.array, mcmc.post_warmup_state)
        chain_warmup_states = split_hmc_state_by_chain(post_warmup_state, n_chains)
        for c in range(n_chains):
            i_chain_ = i_chain + c
            mcmc_seed_ = mcmc_seed + 12345 * i_chain_
            with open(f'{save_base}_{i_chain_}_{mcmc_seed_}_warmup_state.pkl', 'wb') as f:
                pickle.dump(chain_warmup_states[c], f)
        warmup_iter = 0
        # Use proper key splitting inside the warmup loop
        while check_fail:
            warmup_iter += 1
            if warmup_iter > MAX_WARMUP_ITERS:
                logger.error('Reached maximum warmup iterations without passing dense mass matrix check.')
                sys.exit(1)
            rng_key, subkey = jax.random.split(rng_key)
            logger.info(f'Warmup iteration: {warmup_iter}')
            mcmc.warmup(subkey, obs_data=data_1d_ind, extra_fields=('potential_energy',))
            inv_mass_matrix = copy.deepcopy(mcmc.post_warmup_state.adapt_state.inverse_mass_matrix)
            if n_chains == 1:
                inv_mass_matrix[dense_mass[0]] = inv_mass_matrix[dense_mass[0]][None, :, :]
            i_warmup += 1
            index_map, check_fail = check_dense_mass_matrix(inv_mass_matrix, dense_mass[0], criteria)
            logger.info(f'After warmup iteration {i_warmup}: check_fail={check_fail}')
            post_warmup_state = tree_map(jnp.array, mcmc.post_warmup_state)
            chain_warmup_states = split_hmc_state_by_chain(post_warmup_state, n_chains)
            for c in range(n_chains):
                i_chain_ = i_chain + c
                mcmc_seed_ = mcmc_seed + 12345 * i_chain_
                with open(f'{save_base}_{i_chain_}_{mcmc_seed}_warmup_state.pkl', 'wb') as f:
                    pickle.dump(chain_warmup_states[c], f)
        logger.info(f'Final dense mass matrix check passed after {i_warmup} warmup iterations.')

    # -----------------------------
    # MCMC loop: run for i_iter batches
    # -----------------------------
    mean_gauss_1d = [None] * n_chains
    posterior_samples = {param: None for param in params}

    for i in range(i_iter):
        logger.info("running batch %d ...", i)
        # Split rng_key for each MCMC run
        rng_key = jax.random.PRNGKey(mcmc_seed+12*i_chain+1234*i+12345*i_contd)
        rng_key, run_key = jax.random.split(rng_key)
        mcmc.run(run_key,
        #mcmc.run(mcmc.post_warmup_state.rng_key,
                 obs_data=data_1d_ind,
                 extra_fields=('potential_energy',))
        samples = mcmc.get_samples(group_by_chain=True)
        samples['potential_energy'] = mcmc.get_extra_fields(group_by_chain=True)['potential_energy']
        
        # Accumulate posterior samples
        for param in params:
            arr = samples[param].astype(np.float32)  # shape (chains, batch_size)
            if posterior_samples[param] is None:
                posterior_samples[param] = arr
            else:
                posterior_samples[param] = jnp.concatenate([posterior_samples[param], 
                                                            arr], 
                                                            axis=1)
        
        # Update chain-wise summaries (mean IC and MAP)
        for c in range(n_chains):
            chain_key = f'chain_{i_chain + c}'
            # update mean of IC
            if which_ics == 'varied_ics':
                ic_samples = samples['gauss_1d'][c]
                batch_mean = np.mean(ic_samples, axis=0)
                mean_gauss_1d[c] = batch_mean if i == 0 else (i * mean_gauss_1d[c] + batch_mean) / (i + 1)
            # update MAP samples
            pe = samples['potential_energy'][c]
            idx_min = int(jnp.argmin(pe))
            if i_contd == 0 or pe[idx_min] <= map_samples[chain_key]['potential_energy']:
                map_samples[chain_key] = {p: samples[p][c, idx_min] for p in map_params}
                logger.info(f"Updated MAP samples for {chain_key}: {map_samples[chain_key]}")

        last_state = mcmc.last_state
        mcmc.post_warmup_state = last_state

        # Save last state at final batch
        if i == i_iter - 1:
            i_last = i + i_contd + 1
            for c in range(n_chains):
                with open(f'{save_base}_{i_chain + c}_{i_last}_last_state.pkl', 'wb') as f:
                    pickle.dump(split_hmc_state_by_chain(last_state, n_chains)[c], f)
        
        '''
        if i == 0:
            for param in params:
                posterior_samples[param] = jax.device_put(samples[param].astype(np.float32), cpus[0])
            if which_ics == 'varied_ics' and do_pk: ### compute pk_ic
                gauss_1d_stack = jnp.stack([jnp.array(samples['gauss_1d'][c]) for c in range(n_chains)], 
                                           axis=0)
                pk_ic = batch_pk_ic_pmap(gauss_1d_stack,
                                         ng,
                                         boxsize**3,
                                         measure_pk)
                if true_gauss_3d is not None:
                    pk_ic_res_cross = batch_pk_ic_res_cross_pmap(jnp.array(samples['gauss_1d']),
                                                                 jnp.array(true_gauss_3d),
                                                                 ng,
                                                                 boxsize**3,
                                                                 measure_pk)
            for c in range(n_chains):
                i_chain_ = i_chain + c
                if which_ics == 'varied_ics':
                    mean_gauss_1d[c] = np.mean(samples['gauss_1d'][c], axis=0)
                curr_min_idx = np.argmin(samples['potential_energy'][c])
                curr_min_value = samples['potential_energy'][c][curr_min_idx]
                if i_contd == 0:
                    map_samples[f'chain_{i_chain_}'] = {}
                    for param in map_params:
                        map_samples[f'chain_{i_chain_}'][param] = samples[param][c][curr_min_idx]
                else:
                    if curr_min_value <= map_samples[f'chain_{i_chain_}']['potential_energy']:
                        for param in map_params:
                            map_samples[f'chain_{i_chain_}'][param] = samples[param][c][curr_min_idx]
                logger.info(f'MAP samples (initial batch) in chain {i_chain_}:')
                for param in map_params:
                    logger.info(f'{param}: {map_samples[f'chain_{i_chain_}'][param]}')
        else:
            for param in params:
                posterior_samples[param] = np.concatenate(
                    [posterior_samples[param], samples[param].astype(np.float32)], axis=1)
            if which_ics == 'varied_ics' and do_pk: ### compute pk_ic
                gauss_1d_stack = jnp.stack([jnp.array(samples['gauss_1d'][c]) for c in range(n_chains)], 
                                           axis=0)
                pk_ic_batch = batch_pk_ic_pmap(gauss_1d_stack,
                                               ng,
                                               boxsize**3,
                                               measure_pk)
                pk_ic = jnp.concatenate([pk_ic, pk_ic_batch], axis=1)
                if true_gauss_3d is not None:
                    pk_ic_res_cross_batch = batch_pk_ic_res_cross_pmap(jnp.array(samples['gauss_1d']),
                                                                       jnp.array(true_gauss_3d),
                                                                       ng,
                                                                       boxsize**3,
                                                                       measure_pk)
                    pk_ic_res_cross = jnp.concatenate([pk_ic_res_cross, pk_ic_res_cross_batch], axis=1)
            for c in range(n_chains):
                i_chain_ = i_chain + c
                if which_ics == 'varied_ics':
                    batch_mean = np.mean(samples['gauss_1d'][c], axis=0)
                    mean_gauss_1d[c] = (i * mean_gauss_1d[c] + batch_mean) / (i + 1)
                curr_min_idx = np.argmin(samples['potential_energy'][c])
                curr_min_value = samples['potential_energy'][c][curr_min_idx]
                if curr_min_value <= map_samples[f'chain_{i_chain_}']['potential_energy']:
                    for param in map_params:
                        map_samples[f'chain_{i_chain_}'][param] = samples[param][c][curr_min_idx]
                    logger.info(f'MAP samples in chain {i_chain_} updated:')
                    for param in map_params:
                        logger.info(f'{param}: {map_samples[f'chain_{i_chain_}'][param]}')
                    
        last_state = mcmc.last_state
        mcmc.post_warmup_state = last_state
        del samples
        if i == i_iter - 1:
            logger.info("Batch %d completed; saving last state.", i)
            i_last = i + i_contd + 1
            chain_last_states = split_hmc_state_by_chain(last_state, n_chains)
            for c in range(n_chains):
                i_chain_ = i_chain + c
                with open(f'{save_base}_{i_chain_}_{i_last}_last_state.pkl', 'wb') as f:
                    pickle.dump(chain_last_states[c], f)

        '''

    for c in range(n_chains):
        i_chain_ = i_chain + c
        logger.info(f'MAP samples (final) in chain {i_chain_}:')
        for param in map_params:
            logger.info(f'{param}: {map_samples[f'chain_{i_chain_}'][param]}')

    # Combine current posterior_samples with previous samples (if any)
    for c in range(n_chains):
        i_chain_ = i_chain + c
        chain_key = f'chain_{i_chain_}'
        combined_samples = {}
        for param in posterior_samples.keys():
            current = posterior_samples[param][c]  # current samples for chain c
            if chain_key in samples_prev and param in samples_prev[chain_key]:
                combined = np.concatenate([samples_prev[chain_key][param], current], axis=0)
                combined_samples[param] = combined
                logger.info(f'Chain {i_chain_}, parameter {param}: concatenated samples shape = {combined.shape}')
            else:
                combined_samples[param] = current
                logger.info(f'Chain {i_chain_}, parameter {param}: new samples shape = {current.shape}')
        # Save gauss_1d mean (restart-capable)
        if which_ics == 'varied_ics':
            if i_contd > 0:
                factor = i_contd * int(i_sample / thin)
                new_mean = (n_samples * mean_gauss_1d[c] + factor * mean_gauss_1d_prev[chain_key]) / (n_samples + factor)
                logger.info(f'Chain {i_chain_}: Updated gauss_1d mean: new shape = {new_mean.shape}')
            else:
                new_mean = mean_gauss_1d[c]
            save_file = f'{save_base}_samples_chain{i_chain_}.npz'
            save_kwargs = {
                'samples': combined_samples,
                'MAP_samples': map_samples[chain_key],
                'mean_gauss_1d': new_mean,
                }
            np.savez_compressed(save_file, **save_kwargs)
            logger.info(f'Saved combined samples for chain {i_chain_} to {save_file}')
        else:
            save_file = f'{save_base}_samples_chain{i_chain_}.npz'
            np.savez_compressed(save_file, 
                                samples=combined_samples,
                                MAP_samples=map_samples[chain_key])
            logger.info(f'Saved combined samples for chain {i_chain_} to {save_file}')

    if i_contd > 0:
        for c in range(n_chains):
            i_chain_ = i_chain + c
            state_file = f'{save_base}_{i_chain_}_{i_contd}_last_state.pkl'
            if os.path.exists(state_file):
                os.remove(state_file)
                logger.info(f'Removed temporary file {state_file}')
