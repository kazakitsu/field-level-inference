#!/usr/bin/env python3
import os
import sys
import copy
import pickle
import logging
import numpy as np
from typing import NamedTuple, Dict
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random, jit
try:
    tree_map = jax.tree.map
except AttributeError:  # JAX ~ 0.4
    from jax.tree_util import tree_map

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Custom modules
import field_level.idx as idx
import field_level.cosmo_util as cosmo_util
from lss_utils.spectra_util_jax import Measure_Pk
import PT_field
from PT_field.forward_model_jax import LPT_Forward, EPT_Forward

# Logging configuration
logging.basicConfig(stream=sys.stderr,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# JAX configuration
if jax.config.read('jax_enable_x64'):
    numpyro.enable_x64()
    logger.info('NumPyro x64 mode is enabled because JAX is in x64 mode.')
    r_dtype = jnp.float64
    c_dtype = jnp.complex128
else:
    r_dtype = jnp.float32
    c_dtype = jnp.complex64 

logger.info(f'The inference is running on {jax.default_backend()}')

# Constants
DEFAULT_I_SAMPLE = 100
MAX_WARMUP_ITERS = 10  # For preventing infinite warmup loop

LN1010AS_TRUE = 3.047
OMEGA_B_TRUE = 0.02242
OMEGA_C_TRUE = 0.11933
HUBBLE_TRUE = 0.73
NS_TRUE = 0.9665

K_NL = 0.2  # Non-linear scale in h/Mpc, can be adjusted based on cosmology

@dataclass(frozen=True)
class ModelSpec:
    fwd_kind: str       # 'lpt' | 'ept' | 'gauss'
    lpt_order: int      # 0 for non-LPT
    pt_order:  int      # 0 for non-EPT/gauss
    bias_order: int     # 0..3
    rsd: bool
    matter: bool

class Parsed(NamedTuple):
    spec: ModelSpec
    tags: Dict[str, bool]  # reserved for future use

def parse_model_name(model_name: str) -> Parsed:
    name = model_name.lower()
    rsd    = 'rsd' in name
    matter = 'matter' in name

    # bias order
    if   'cubic' in name: bias_order = 3
    elif 'quad'  in name: bias_order = 2
    elif 'lin'   in name: bias_order = 1
    else:                 bias_order = 0

    # forward kind & order
    if 'lpt' in name:
        fwd_kind = 'lpt'
        lpt_order = 1 if '1lpt' in name else (2 if '2lpt' in name else 1)
        pt_order  = 0
    elif 'ept' in name or 'gauss' in name:
        fwd_kind = 'ept' if 'ept' in name else 'gauss'
        pt_order  = 1 if ('1ept' in name or 'gauss' in name) else (2 if '2ept' in name else 1)
        lpt_order = 0
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    spec = ModelSpec(
        fwd_kind=fwd_kind, lpt_order=lpt_order, pt_order=pt_order,
        bias_order=bias_order, rsd=rsd, matter=matter,
    )
    return Parsed(spec=spec, tags={})

def build_forward_model(spec: ModelSpec, *, boxsize, ng, ng_L, ng_E, mas_params, r_dtype):
    if spec.fwd_kind == 'lpt':
        return LPT_Forward(
            boxsize=boxsize, ng=ng, ng_L=ng_L, ng_E=ng_E,
            mas_cfg=mas_params, rsd=spec.rsd, lpt_order=spec.lpt_order,
            bias_order=spec.bias_order, dtype=r_dtype
        )
    else:
        return EPT_Forward(
            boxsize=boxsize, ng=ng, ng_L=ng_L, ng_E=ng_E,
            rsd=spec.rsd, pt_order=spec.pt_order,
            bias_order=spec.bias_order, dtype=r_dtype
        )

# -----------------------------
# Helper Functions
# -----------------------------

def make_run_key(base_seed: int, *, chain_id: int, batch_id: int, resume_iter: int) -> jax.random.PRNGKey: 
    key = random.PRNGKey(base_seed) 
    key = random.fold_in(key, chain_id)    # independent for each chain 
    key = random.fold_in(key, resume_iter) # independent for each restart 
    key = random.fold_in(key, batch_id)    # independent for each batch within a chain 
    return key

def make_run_keys(base_seed: int,  *, chain_ids, batch_id: int, resume_iter: int):
    keys = []
    for cid in chain_ids:
        key = random.PRNGKey(base_seed)
        key = random.fold_in(key, cid)          # independent for each chain
        key = random.fold_in(key, resume_iter)  # independent for each restart
        key = random.fold_in(key, batch_id)     # independent for each batch within a chain
        keys.append(key)
    return jnp.stack(keys, axis=0)             # (n_chains, 2)

def _is_scalar_like(x):
    # Return True for Python/NumPy/JAX scalars
    return isinstance(x, (int, float, np.floating)) or (
        isinstance(x, (np.ndarray, jnp.ndarray)) and np.asarray(x).ndim == 0
    )

def _seq_len(cfg):
    """
    Normalize config length:
      - None -> 0
      - scalar -> 1
      - sequence -> len(cfg)
    """
    if cfg is None:
        return 0
    if _is_scalar_like(cfg):
        return 1
    try:
        return len(cfg)
    except TypeError:
        return 1

def _as_scalar(cfg):
    """
    Return a scalar value:
      - if cfg is a scalar, return it
      - if cfg is a 1-element sequence, return cfg[0]
      - otherwise the caller should handle (e.g., len>=2)
    """
    if _is_scalar_like(cfg):
        return float(cfg)
    return float(cfg[0])

def _is_free(cfg):
    """Two numbers => treat as free (sampled); one number (or scalar) => fixed."""
    return _seq_len(cfg) == 2

def _det_if(name, value, cond):
    """Record as deterministic only when cond=True."""
    return numpyro.deterministic(name, value) if cond else value

def _resolve_pair_scaled_raw(
    params_dict,
    scaled_key,              # e.g., "A2b2"
    raw_key,                 # e.g., "b2"
    factor,                  # e.g., A**2
    draw_scaled,             # e.g., draw_normal_or_fix
    draw_raw,                # e.g., draw_normal_or_fix / draw_uniform_or_fix
    default_raw,             # default for raw when neither key is provided
    record_derived_if_sampled=True,
):
    """
    Resolve a linked parameter pair (scaled_key <-> factor * raw_key).

    Priority (keeps your current behavior):
      - If scaled_key is present:
          * free  -> sample scaled; raw = deterministic(scaled/factor)
          * fixed -> scaled=const;  raw = scaled/factor         (no deterministic)
      - elif raw_key is present:
          * free  -> sample raw;    scaled = deterministic(factor*raw)
          * fixed -> raw=const;     scaled = factor*raw         (no deterministic)
      - else:
          * raw = default_raw; scaled = factor*raw              (no deterministic)

    Returns (raw_value, scaled_value).
    """
    if scaled_key in params_dict:
        cfg = params_dict[scaled_key]
        if _is_free(cfg):
            scaled = draw_scaled(scaled_key, cfg)
            raw    = _det_if(raw_key, scaled / factor, record_derived_if_sampled)
        else:
            scaled = _as_scalar(cfg)
            raw    = scaled / factor
    elif raw_key in params_dict:
        cfg = params_dict[raw_key]
        if _is_free(cfg):
            raw    = draw_raw(raw_key, cfg)
            scaled = _det_if(scaled_key, factor * raw, record_derived_if_sampled)
        else:
            raw    = _as_scalar(cfg)
            scaled = factor * raw
    else:
        raw    = default_raw
        scaled = factor * raw
    return raw, scaled


def draw_uniform_or_fix(name, cfg):
    """Accepts scalar or [x] or [lo, hi]; samples Uniform for len==2, otherwise fixes the value."""
    n = _seq_len(cfg)
    if n == 0:
        return 0.0
    if n == 2:
        lo, hi = cfg
        return numpyro.sample(name, dist.Uniform(float(lo), float(hi)))
    if n >= 1:
        return _as_scalar(cfg)
    raise ValueError(f"Bad prior spec for {name}: {cfg}")

def draw_normal_or_fix(name, cfg):
    """Accepts scalar or [x] or [mean, std]; samples Normal for len==2, otherwise fixes the value."""
    n = _seq_len(cfg)
    if n == 0:
        return 0.0
    if n == 2:
        mean, std = cfg
        return numpyro.sample(name, dist.Normal(float(mean), float(std)))
    if n >= 1:
        return _as_scalar(cfg)
    raise ValueError(f"Bad prior spec for {name}: {cfg}")

def sample_uniform_deterministic_or_fix(param_name, scaled_name, cfg, default_val):
    """
    Sample Uniform(min,max) with a deterministic record (when len==2),
    or use a fixed value (scalar or len==1), or fall back to `default_val` if cfg is None.
    """
    n = _seq_len(cfg)
    if n == 2:
        lo, hi = cfg
        return sample_uniform_deterministic(param_name, scaled_name, (float(lo), float(hi)))
    if n >= 1:
        return _as_scalar(cfg)
    return default_val


def sample_uniform_deterministic(param_name, scaled_name, bounds):
    """
    Sample a value from a Uniform distribution and record it as a deterministic value.
    """
    min_val, max_val = bounds
    scaled = numpyro.sample(scaled_name, dist.Uniform(0.0, 1.0))
    value = numpyro.deterministic(param_name, min_val + (max_val - min_val) * scaled)
    return value

def _normalize_dense_mass_groups(dense_mass, cosmo_params):
    """
    Normalize user-provided dense_mass groups before passing to NUTS.

    Rules:
      - For oc/ob/hubble/ns:
        * If prior is a 2-tuple (we sample it), rename to 'scaled_<name>'.
        * Otherwise (fixed), drop it from dense mass (no sample site exists).
      - Other names are kept as-is.

    Returns a list of tuple groups with empty groups removed and duplicates deduped.
    """
    if not dense_mass:
        return dense_mass

    rename_targets = ("oc", "ob", "hubble", "ns", "sigma8")

    def _is_uniform_two_tuple(cfg):
        return _seq_len(cfg) == 2

    def _map_name(name):
        if name in rename_targets:
            cfg = cosmo_params.get(name, None)
            if _is_uniform_two_tuple(cfg):
                return f"scaled_{name}"   # this sample site exists
            else:
                return None               # fixed => no sample site; drop
        return name  # keep others untouched

    normalized = []
    for group in dense_mass:
        # Accept tuple/list; map and drop Nones; also remove duplicates while preserving order
        seen = set()
        mapped = []
        for p in group:
            newp = _map_name(p)
            if (newp is not None) and (newp not in seen):
                mapped.append(newp)
                seen.add(newp)
        if mapped:
            normalized.append(tuple(mapped))

    return normalized

def check_dense_mass_matrix(inv_mass_matrix, dense_mass, criteria=0.9):
    """
    Check each parameter (except 'c0', 'c2', 'c4', 'Sigma2', 'Sigma2_mu2', 'Sigma2_mu4') in the mass matrix block corresponding to dense_mass.
    Returns an index mapping and a boolean indicating if any diagonal value exceeds the criteria.
    """
    block = inv_mass_matrix[dense_mass]
    check_fail = False
    index_map = {}
    for i, param in enumerate(dense_mass):
        if param in ['c0', 'c2', 'c4', 'Sigma2', 'Sigma2_mu2', 'Sigma2_mu4']:
            continue
        index_map[param] = i
        diag_val = jnp.abs(block[:, i, i])
        if (diag_val > criteria).any():
            logger.info(f'Parameter {param} (index {i}) has diag value {diag_val} exceeding criteria {criteria}')
            check_fail = True
        else:
            logger.info(f'Parameter {param} (index {i}) OK: diag value {diag_val}')
    return index_map, check_fail

def tree_block_until_ready(x):
    return tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)

def tree_to_host(x):
    return tree_map(lambda a: np.asarray(a) if isinstance(a, (jnp.ndarray, np.ndarray)) else a, x)

def tree_to_device(x):
    return tree_map(lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, x)

def split_hmc_state_by_chain(hmc_state, num_chains):
    if num_chains == 1:
        return [hmc_state]
    def select(a, i):
        return a[i] if (hasattr(a, "shape") and a.shape[:1] == (num_chains,)) else a
    return [tree_map(lambda a, i=i: select(a, i), hmc_state) for i in range(num_chains)]

def merge_hmc_states(per_chain_states):
    return tree_map(lambda *xs: jnp.stack(xs, axis=0), *per_chain_states)

def _set_rng_key_on_state(state, key):
    # numpyro.infer.mcmc.HMCState is NamedTuple
    if hasattr(state, "_replace") and hasattr(state, "rng_key"):
        return state._replace(rng_key=key)
    try:
        d = state._asdict()
        d["rng_key"] = key
        return type(state)(**d)
    except Exception:
        return state
    
def broadcast_state_to_chains(template_state,
                              n_chains: int,
                              *,
                              base_seed: int,
                              resume_iter: int,
                              chain_id_start: int):
    """
    Create a batched HMCState for `n_chains` from a single-chain template.
    Only the rng_key differs per chain; all other fields are copied.
    """
    per = []
    for c in range(n_chains):
        cid = chain_id_start + c
        key = jax.random.PRNGKey(base_seed)
        key = jax.random.fold_in(key, resume_iter)
        key = jax.random.fold_in(key, cid)
        st_c = _set_rng_key_on_state(template_state, key)
        per.append(st_c)
    return merge_hmc_states(per)

def save_hmc_state(state, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tree_to_host(state), f, protocol=pickle.HIGHEST_PROTOCOL)

def load_hmc_state(path: str):
    with open(path, "rb") as f:
        return tree_to_device(pickle.load(f))

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

def save_samples_npz(save_file: str,
                     samples: dict,
                     map_samples: dict,
                     mean_gauss_1d: np.ndarray | None):
    arrays = {k: np.asarray(v) for k, v in samples.items()}
    arrays.update({f"MAP_{k}": np.asarray(v) for k, v in map_samples.items()})
    if mean_gauss_1d is not None:
        arrays["mean_gauss_1d"] = np.asarray(mean_gauss_1d)
    with open(save_file, 'wb') as f:
        np.savez_compressed(f, **arrays)

def load_samples_npz(save_file: str):
    z = np.load(save_file)
    arrays = {k: z[k] for k in z.files}
    samples = {k: v for k, v in arrays.items() if not (k.startswith("MAP_") or k == "mean_gauss_1d")}
    map_samples = {k[4:]: arrays[k] for k in arrays.keys() if k.startswith("MAP_")}
    mean_gauss_1d = arrays.get("mean_gauss_1d", None)
    return samples, map_samples, mean_gauss_1d

def sanity_check_restart_counts(save_base: str,
                                chain_ids,
                                params,
                                i_contd: int,
                                i_sample: int):
    """
    Verify, on restart (i_contd > 0), that the already-saved .npz files
    contain exactly N_prev = i_contd * i_sample post-thinned samples per chain.

    We only check parameters that:
      - are listed in `params` we plan to keep saving, and
      - are present in the .npz file, and
      - have at least 1 dimension (sample axis).
    If any chain fails the check, raise ValueError with details.
    """
    expected = int(i_contd * i_sample)
    problems = []

    for cid in chain_ids:
        fn = f"{save_base}_samples_chain{cid}.npz"
        if not os.path.exists(fn):
            problems.append((cid, "missing_file", fn))
            continue
        try:
            z = np.load(fn)
        except Exception as e:
            problems.append((cid, f"failed_to_open: {e}", fn))
            continue

        # Collect lengths along the first axis for all checkable params
        lengths = {}
        for p in params:
            if p in z.files:
                arr = z[p]
                if hasattr(arr, "shape") and arr.ndim >= 1:
                    lengths[p] = int(arr.shape[0])
        z.close()

        # If nothing to check, flag (could happen if nearly nothing was saved)
        if not lengths:
            problems.append((cid, "no_checkable_params", fn))
            continue

        uniq = set(lengths.values())
        if len(uniq) != 1 or expected not in uniq:
            problems.append((cid, {"expected": expected, "found": lengths}, fn))

    if problems:
        lines = ["Restart sanity check failed:"]
        for cid, what, fn in problems:
            lines.append(f"  chain {cid}: {what} @ {fn}")
        lines.append("Hint: i_contd is the number of completed batches; "
                     "each batch persists i_sample post-thinned samples per parameter.")
        raise ValueError("\n".join(lines))

    logger.info("Restart sanity check passed: found %d samples in each chain.", expected)


def _log_map_update(chain_key: str,
                    map_dict: dict,
                    *,
                    preview_elems: int = 8,
                    whitelist: tuple | list | None = None,
                    blacklist: tuple | list | None = ('gauss_1d',)):
    """
    Log MAP sample values in a readable and safe way.

    Rules:
      - Scalars (ndim == 0): print the numeric value.
      - Small arrays (size <= preview_elems): print full values.
      - Large arrays: print shape and the first `preview_elems` elements only.

    You can pass `whitelist` to restrict which keys to log, or `blacklist`
    to skip noisy keys (by default, 'gauss_1d' is skipped).
    """
    keys = list(map_dict.keys())
    if whitelist is not None:
        keys = [k for k in keys if k in whitelist]
    if blacklist is not None:
        keys = [k for k in keys if k not in blacklist]

    for k in keys:
        v = map_dict[k]
        a = np.asarray(v)
        if a.ndim == 0:
            logger.info("[MAP %s] %s = %.6g", chain_key, k, float(a))
        else:
            flat = a.reshape(-1)
            if flat.size <= preview_elems:
                logger.info("[MAP %s] %s shape=%s values=%s", chain_key, k, a.shape, flat)
            else:
                head = flat[:preview_elems]
                logger.info("[MAP %s] %s shape=%s head=%s ...", chain_key, k, a.shape, head)


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
        The cosmological parameters in the forward model. The keys should be
            'A' or 'sigma8': the amplitude of the density
            'oc' : the physical density of CDM, oc = Omega_cdm * h^2
            'hubble' : the hubble parameter
            'ob' : the physical density of baryon, ob = Omega_b * h^2
            'ns' : the spectral index of the primordial power spectrum
        Each value should be a tuple of (min, max) for prior, or a single value to fix it.
    
    bias_params : dict
        The bias parameters in the foward model, including the counter terms. The keys should be
            'b1': the bias to \delta
            'b2': the bias to \delta^2
            'bG2': the bias to G2
            'bGamma3': the bias to Gamma3
            'c0': the coefficient to k^2 \delta
            'c2' : the coefficient to k^2 \mu^2 \delta
            'c4' : the coefficient to k^2 \mu^4 \delta
            'Sigma2': the coefficient in exp(-0.5 k^2 \Sigma2)
            'Sigma2_mu2' : the coefficient in exp(-0.5 k^2 \mu^2 \Sigma2)
            'Sigma2_mu4' : the coefficient in exp(-0.5 k^2 \mu^4 \Sigma2)
        Each value should be a tuple of (min, max) or (mean, std) for prior, or a single value to fix it.

    err_params : dict
        The error (in the likelihood) parameters to sample (or not). The keys should be
            'log_Perr' : The logarithm of the (white) noise power spectrum
        The value of 'log_Perr' should (mean, std) for prior, or a single value to fix it.
    
    kmax : float
        The maximum k used in the likelihood.
        If kmax > 1.0 the cubic cutoff with the kmax^3 grid is used, instead of the default spherical cutoff.
    
    dense_mass : [tuple]
        The parameters whose mass matrix is full-rank.
        
    mcmc_params : (int, int, int, int, int, float, int, int)
        (i-th chain, # of chains, thinning, # of samples, # of warmup, target acceptance rate, random seed for mcmc, # of the previously collected samples (to restart) )
        # of chains can be greater than 1 only if i-th chain < 0.

    """
    which_ics, collect_ics = ics_params
    logger.info(f'Initial IC settings: {which_ics}')
    window_order, interlace = mas_params
    i_chain, n_chains, thin, n_samples, n_warmup, accept_rate, mcmc_seed, i_contd = mcmc_params
    #numpyro.set_host_device_count(n_chains)
    if i_contd > 0:
        n_warmup = 0

    assert 'log_Perr' in err_params, "err_params['log_Perr'] is required"

    vol = jnp.asarray(boxsize, dtype=r_dtype)**3

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
    ng, ng_L, ng_E = map(int, ng_params)
    logger.info("ng = %d, ng_L = %d, ng_E = %d, kmax = %s", ng, ng_L, ng_E, kmax)

    ng3 = int(ng**3)

    ### flags for pk_ic computations
    if isinstance(pk_params, (list, tuple)) and len(pk_params)==3:
        pk_nbin, pk_kmin, pk_kmax = pk_params
        do_pk = pk_nbin is not None
    else:
        do_pk = False

    if do_pk:
        kbin_edges = jnp.linspace(pk_kmin, pk_kmax, pk_nbin + 1, dtype=r_dtype)
        measure_pk = Measure_Pk(boxsize, ng, kbin_edges, ell_max=0, dtype=r_dtype)
        dummy = jnp.ones((ng, ng, ng // 2 + 1)) + 0.*1j
        dummy_out = measure_pk(dummy)
        dummy_out = measure_pk(dummy, dummy)

    # Create the forward model
    parsed = parse_model_name(model_name)
    spec   = parsed.spec
    logger.info(f'Parsed model: {spec}')
    fwd_model = build_forward_model(spec,
                                    boxsize=boxsize, ng=ng, ng_L=ng_L, ng_E=ng_E, 
                                    mas_params=mas_params, r_dtype=r_dtype)

    # prepare the linear P(k) generator
    pk_lin_gen = cosmo_util.Pk_Provider()

    try:
        # Pre-build the k-caches
        fwd_model._ensure_k_caches()

        cosmo_params_ = jnp.array([OMEGA_B_TRUE, OMEGA_C_TRUE, HUBBLE_TRUE, NS_TRUE, LN1010AS_TRUE, 0.0])
        pk_lin_table_ = pk_lin_gen.pk_lin_table(cosmo_params_)

        # Pre-build interpolation caches
        g0_  = jnp.ones((ng, ng, ng//2 + 1), dtype=fwd_model.complex_dtype)
        _ = fwd_model.linear_modes(pk_lin_table_, g0_).block_until_ready()
    except Exception as e:
        logger.warning("Pre-building caches failed: %s", e)                             
    
    gauss_1d_to_3d = idx.build_gauss_1d_to_3d(ng)

    def build_batch_ic_pk_fns(measure_pk, gauss_1d_to_3d, vol, true_gk=None):
        """Return compiled fns to compute IC P(k) on (batch, ng^3) samples.
        If true_gk is provided (k-space complex rfftn layout), also compute cross and residual.
        Outputs:
            auto_fn(batch, ng3) -> (batch, 2, nbin)  where axis0=[k, P]
            xres_fn (optional):  -> tuple of 3 arrays (auto, cross, resid), each (batch, 2, nbin)
        """
        def _to_k(gauss_1d):
            # Convert 1D Gaussian to complex k-space field (rfftn layout) and scale
            gk = gauss_1d_to_3d(gauss_1d) / jnp.sqrt(2.0 * vol)
            return gk

        def _kp(out_mat):
            # out_mat: (nbin, 3) with columns [k_mean, Pk*vol, Nk]
            # Return (2, nbin) stacking k and P only
            k = out_mat[:, 0]
            P = out_mat[:, 1]
            return jnp.stack([k, P], axis=0)

        def _one_auto(gauss_1d):
            gk = _to_k(gauss_1d)
            out = measure_pk(gk)          # (nbin, 3)
            return _kp(out)               # (2, nbin)

        auto_fn = jax.jit(jax.vmap(_one_auto, in_axes=0, out_axes=0))  # (batch, 2, nbin)

        if true_gk is None:
            return auto_fn, None

        def _one_all(gauss_1d):
            gk = _to_k(gauss_1d)
            out_auto  = measure_pk(gk)            # (nbin, 3)
            out_cross = measure_pk(gk, true_gk)   # (nbin, 3)
            resid = gk - true_gk
            out_resid = measure_pk(resid)         # (nbin, 3)
            A = _kp(out_auto)
            C = _kp(out_cross)
            R = _kp(out_resid)
            return (A, C, R)

        xres_fn = jax.jit(jax.vmap(_one_all, in_axes=0, out_axes=(0, 0, 0)))
        return auto_fn, xres_fn

    true_gk = None
    if do_pk and which_ics == 'varied_ics' and (true_gauss_3d is not None):
        # true_gauss_3d must already be in k-space complex rfftn layout
        true_gk = jnp.asarray(true_gauss_3d, dtype=c_dtype) / jnp.sqrt(2.0 * vol)

    if do_pk and which_ics == 'varied_ics':
        auto_pk_fn, xres_pk_fn = build_batch_ic_pk_fns(measure_pk, gauss_1d_to_3d, vol, true_gk=true_gk)
    else:
        auto_pk_fn, xres_pk_fn = None, None

    # --- Compute indices & build extractor for independent modes (k_space) ---
    if which_space == 'k_space':
        if kmax > 1.0:
            ng_max = int(kmax)
            if ng < ng_max:
                raise ValueError(f'kmax={kmax} requires ng >= {ng_max}')
            idx_re, idx_im = idx.indep_modes_kmax_indices(ng_max, kmax=None)
        else:
            ng_max = ng
            idx_re, idx_im = idx.indep_modes_kmax_indices(
                ng_max, kx=fwd_model.kx_, ky=fwd_model.ky_, kz=fwd_model.kz_, kmax=float(kmax)
            )        
    elif which_space == 'r_space':
        kmax = int(kmax)
        ng_max = kmax
        if ng < ng_max:
            raise ValueError(f'kmax={kmax} requires ng >= {ng_max}')
        ng3_max = ng_max ** 3

    @jit
    def independent_modes(fieldk):
        """
        Return independent modes as a 1D vector:
        [Re kept..., Im kept...], order matches ravel() scan order.
        """
        fk = fieldk.reshape(-1)
        re = jnp.take(fk.real, idx_re)
        im = jnp.take(fk.imag, idx_im)
        return jnp.hstack([re, im])

    # Load the data
    data = load_data(data_path)

    # Preprocess data to 1D (depending on space)
    if which_space == 'k_space':
        data[0,0,0] = 0.0  # Remove DC mode
        if ng_max < data.shape[0]:
            data_max = PT_field.func_reduce(ng_max, data)
        elif ng_max > data.shape[0]:
            logger.warning(f'Input data size {data.shape[0]} < ng {ng}; padding with zeros.')     
            data_max = PT_field.func_extend(ng_max, data)
        else:
            data_max = data
        data_1d_ind = independent_modes(data_max)
    elif which_space == 'r_space':
        if data.shape[2] != data.shape[0]:
            logger.info('Data is in Fourier space.')
            data[0,0,0] = 0.0
            data_ = PT_field.func_reduce(ng_max, data)
            datar = jnp.fft.irfftn(data_, norm='forward')
            data_1d_ind = datar.reshape(ng_max**3)
        else:
            if data.shape[0] != ng_max:
                data_k = jnp.fft.rfftn(data, norm='forward')
                data_k = PT_field.func_reduce(ng_max, data_k)
                data = jnp.fft.irfftn(data_k, norm='forward')
            data_1d_ind = data.reshape(ng_max**3)
            data_1d_ind -= data_1d_ind.mean()
        logger.info('data_r_mean = %s', data_1d_ind.mean())
    data_1d_ind = jnp.asarray(data_1d_ind, dtype=r_dtype)

    logger.info('data_1d_ind.shape = %s', data_1d_ind.shape)

    # -----------------------------
    # Model definition
    # -----------------------------
    
    def model(obs_data):
        if which_ics == 'varied_ics':
            gauss_1d = numpyro.sample("gauss_1d", dist.Normal(0.0, 1.0), sample_shape=(ng3,))
            gauss_3d = gauss_1d_to_3d(gauss_1d)
        else:
            gauss_3d = jnp.asarray(true_gauss_3d, dtype=c_dtype)

        if 'cosmo' in which_pk:
            omega_c = sample_uniform_deterministic_or_fix('oc',     'scaled_oc',     cosmo_params.get('oc',     None), OMEGA_C_TRUE)
            hubble  = sample_uniform_deterministic_or_fix('hubble', 'scaled_hubble', cosmo_params.get('hubble', None), HUBBLE_TRUE)
            omega_b = sample_uniform_deterministic_or_fix('ob',     'scaled_ob',     cosmo_params.get('ob',     None), OMEGA_B_TRUE)
            ns      = sample_uniform_deterministic_or_fix('ns',     'scaled_ns',     cosmo_params.get('ns',     None), NS_TRUE)

            oc_cfg = cosmo_params.get('oc', None)
            ob_cfg = cosmo_params.get('ob', None)
            h_cfg  = cosmo_params.get('hubble', None)

            any_free = any(_is_free(cfg) for cfg in (oc_cfg, ob_cfg, h_cfg))
            OM_val = (omega_b + omega_c) / hubble**2
            OM = _det_if('OM', OM_val, any_free)

            if _is_free(h_cfg):
                H0 = numpyro.deterministic('H0', 100.0 * hubble)
            else:
                H0 = 100.0 * hubble

            if 'ln1010As' in cosmo_params:
                ln1010As = cosmo_params['ln1010As'][0]
            else:
                ln1010As = LN1010AS_TRUE

            cosmo_params_local = jnp.array([omega_b, omega_c, hubble, ns, ln1010As, 0.0])
            pk_lin_table_ = pk_lin_gen.pk_lin_table(cosmo_params_local)

            if 'sigma8' in cosmo_params:
                cfg = cosmo_params['sigma8']
                n = _seq_len(cfg)
                if n == 2:
                    lo, hi = cfg
                    scaled_sigma8 = numpyro.sample('scaled_sigma8', dist.Uniform(0.0, 1.0))
                    sigma8 = numpyro.deterministic('sigma8', float(lo) + (float(hi) - float(lo)) * scaled_sigma8)
                else:
                    sigma8 = numpyro.deterministic('sigma8', _as_scalar(cfg))
                tmp_sigma8 = pk_lin_gen.sigmaR(pk_lin_table_)
                A = numpyro.deterministic('A', sigma8 / tmp_sigma8)

            elif 'A' in cosmo_params:
                cfg = cosmo_params['A']
                n = _seq_len(cfg)
                if n == 2:
                    lo, hi = cfg
                    A = numpyro.sample('A', dist.Uniform(float(lo), float(hi)))
                    sigma8 = numpyro.deterministic('sigma8', A * pk_lin_gen.sigmaR(pk_lin_table_))
                else:
                    A = _as_scalar(cfg)

            else:
                A = 1.0
    
            cosmo_params_local = cosmo_params_local.at[-1].set(redshift)
            pk_lin_table = pk_lin_gen.pk_lin_table(cosmo_params_local)

        A2 = A * A
        A3 = A * A * A

        # Bias parameters sampling
        # (Ab1, b1): uniform; default b1=1.0
        b1, Ab1 = _resolve_pair_scaled_raw(
            bias_params, "Ab1", "b1", A,
            draw_scaled=draw_uniform_or_fix,
            draw_raw=draw_uniform_or_fix,
            default_raw=1.0,
        )

        # (A2b2, b2): normal; default b2=0.0
        b2, A2b2 = _resolve_pair_scaled_raw(
            bias_params, "A2b2", "b2", A2,
            draw_scaled=draw_normal_or_fix,
            draw_raw=draw_normal_or_fix,
            default_raw=0.0,
        )

        # (A2bG2, bG2): special-case for 'G2' models
        if 'G2' in model_name:
            bG2  = b1
            A2bG2 = A2 * bG2
        else:
            bG2, A2bG2 = _resolve_pair_scaled_raw(
                bias_params, "A2bG2", "bG2", A2,
                draw_scaled=draw_normal_or_fix,
                draw_raw=draw_normal_or_fix,
                default_raw=0.0,
            )
        
        # (A3bGamma3, bGamma3): normal; default bGamma3=0.0
        bGamma3, A3bGamma3 = _resolve_pair_scaled_raw(
            bias_params, "A3bGamma3", "bGamma3", A3,
            draw_scaled=draw_normal_or_fix,
            draw_raw=draw_normal_or_fix,
            default_raw=0.0,
        )

        # (A3b3, b3): normal; default b3=0.0
        b3, A3b3 = _resolve_pair_scaled_raw(
            bias_params, "A3b3", "b3", A3,
            draw_scaled=draw_normal_or_fix,
            draw_raw=draw_normal_or_fix,
            default_raw=0.0,
        )

        # (A3bG2d, bG2d): normal; default bG2d=0.0
        bG2d, A3bG2d = _resolve_pair_scaled_raw(
            bias_params, "A3bG2d", "bG2d", A3,
            draw_scaled=draw_normal_or_fix,
            draw_raw=draw_normal_or_fix,
            default_raw=0.0,
        )
        
        # (A3bG3, bG3): normal; default bG3=0.0
        bG3, A3bG3 = _resolve_pair_scaled_raw(
            bias_params, "A3bG3", "bG3", A3,
            draw_scaled=draw_normal_or_fix,
            draw_raw=draw_normal_or_fix,
            default_raw=0.0,
        )

        # Counter terms
        c0 = draw_normal_or_fix('c0', bias_params.get('c0', 0.0))
        c2 = draw_normal_or_fix('c2', bias_params.get('c2', 0.0))
        c4 = draw_normal_or_fix('c4', bias_params.get('c4', 0.0))

        Sigma2      = draw_normal_or_fix('Sigma2',      bias_params.get('Sigma2', 0.0))
        Sigma2_mu2  = draw_normal_or_fix('Sigma2_mu2',  bias_params.get('Sigma2_mu2', 0.0))
        Sigma2_mu4  = draw_normal_or_fix('Sigma2_mu4',  bias_params.get('Sigma2_mu4', 0.0))

        # Collect all bias parameters
        if 'lin' in model_name:
            betas = [b1,]
        elif 'quad' in model_name:
            betas = [b1, 0.5*b2, bG2,]
        elif 'cubic' in model_name:
            betas = [b1, 0.5*b2, bG2, b3, bG2d, bG3, bGamma3,]
        else:
            betas = []

        if isinstance(fwd_model, LPT_Forward):
            if 'matter' in model_name:
                betas.insert(0, 1.0)
            else:
                betas.insert(0, 0.0)
        
        if 'gauss' in model_name:
            betas.insert(0, 1.0)

        if 'rsd' in model_name:
            growth_f = numpyro.deterministic('growth_f', cosmo_util.growth_f_fitting(redshift, OM))
            if 'gauss_rsd' in model_name:
                c1 = growth_f
            else:
                c1 = 0.0
        else:
            growth_f = 0.0
            c1 = 0.0

        betas += [c0, c1, c2, c4]
        betas = jnp.asarray(betas, dtype=r_dtype)
        #print('betas.shape = ', betas.shape, file=sys.stderr)

        # Error model
        if 'log_Perr' in err_params:
            cfg = err_params['log_Perr']
            n = _seq_len(cfg)
            if n == 2:
                mean, std = float(cfg[0]), float(cfg[1])
                log_Perr = numpyro.sample("log_Perr", dist.Normal(mean, std))
                Perr = jnp.exp(log_Perr)
            elif n == 1:
                val = _as_scalar(cfg)
                Perr = jnp.exp(val)
            else:
                raise ValueError("err_params['log_Perr'] is required")

        if which_space == 'k_space':
            sigma_err = jnp.sqrt(Perr / (2.0 * vol))
        elif which_space == 'r_space':
            sigma2_err = Perr * ng3_max / vol

        if 'log_Peded' in err_params: ### only in r_space
            cfg = err_params['log_Peded']
            n = _seq_len(cfg)
            if n == 2:
                mean, std = float(cfg[0]), float(cfg[1])
                log_Peded = numpyro.sample("log_Peded", dist.Normal(mean, std))
                Peded = numpyro.deterministic("Peded", jnp.exp(log_Peded))
            elif n == 1:
                Peded = jnp.exp(_as_scalar(cfg))
            else:
                raise ValueError("Bad err_params['log_Peded']")

            bound = jnp.sqrt(Perr * Peded)
            sigma2_eded = Peded * ng3_max / vol

            scaled_Peed = numpyro.sample("scaled_Peed", dist.Uniform(-1.0, 1.0))
            Peed = numpyro.deterministic("Peed", scaled_Peed * bound)
            ratio = numpyro.deterministic("ratio", Peed / Peded)

            sigma2_eed = Peed * ng3_max / vol

        delk = A * fwd_model.linear_modes(pk_lin_table, gauss_3d)
        delk_L = PT_field.func_extend(ng_L, delk)

        if isinstance(fwd_model, LPT_Forward):
            fields_k_E = fwd_model.get_shifted_fields(delk_L, growth_f=growth_f)
        if isinstance(fwd_model, EPT_Forward):
            fields_k_E = fwd_model.get_fields(delk_L)
        field_k_E = fwd_model.get_final_field(fields_k_E, betas, beta_type='const')
        field_k_E = field_k_E.at[0,0,0].set(0.0)

        if which_space == 'k_space':
            need_sigma = any(k in bias_params for k in ('Sigma2', 'Sigma2_mu2', 'Sigma2_mu4'))
            if need_sigma:
                kx2E = fwd_model.kx2E[:, None, None]
                ky2E = fwd_model.ky2E[None, :, None]
                kz2E = fwd_model.kz2E[None, None, :]
            if 'Sigma2_mu4' in bias_params:
                k2E = kx2E + ky2E + kz2E
                # Avoid division by zero at DC: set factor=1 where k2==0
                factor = jnp.where(k2E > 0.0,
                                   jnp.exp(-0.5 * Sigma2_mu4 * (kz2E * kz2E) / k2E),
                                   1.0)
                field_k_E = field_k_E * factor
            if 'Sigma2_mu2' in bias_params:
                field_k_E = field_k_E * jnp.exp(-0.5 * Sigma2_mu2 * kz2E)
            if 'Sigma2' in bias_params:
                field_k_E = field_k_E * jnp.exp(-0.5 * Sigma2 * kx2E)
                field_k_E = field_k_E * jnp.exp(-0.5 * Sigma2 * ky2E)
                field_k_E = field_k_E * jnp.exp(-0.5 * Sigma2 * kz2E)
            if ng_max < field_k_E.shape[0]:
                field_k_max = PT_field.func_reduce(ng_max, field_k_E)
            elif ng_max > field_k_E.shape[0]:
                field_k_max = PT_field.func_extend(ng_max, field_k_E)
            else:
                field_k_max = field_k_E
            field_1d_ind = independent_modes(field_k_max)
        elif which_space == 'r_space':
            field_k_max = PT_field.func_reduce(ng_max, field_k_E)
            field_r_max = jnp.fft.irfftn(field_k_max, norm='forward')
            field_1d_ind = field_r_max.reshape(ng3_max)
            if 'log_Peded' in err_params:
                delk_max = PT_field.func_reduce(ng_max, delk_L)
                delr_max = jnp.fft.irfftn(delk_max, norm='forward')
                delr_1d_ind = delr_max.reshape(ng3_max)
                d2r_1d_ind = delr_1d_ind * delr_1d_ind
                sigma2_err = sigma2_err + 2.0 * sigma2_eed * delr_1d_ind + sigma2_eded * d2r_1d_ind
            sigma_err = jnp.sqrt(sigma2_err)
        data = numpyro.sample('data', dist.Normal(field_1d_ind, sigma_err), obs=obs_data)

    # -----------------------------
    # Setup before MCMC execution
    # -----------------------------
    dense_mass = _normalize_dense_mass_groups(dense_mass, cosmo_params)
    logger.info('dense_mass (scaled) = %s', dense_mass)
    n_total = thin * n_samples
    i_sample = DEFAULT_I_SAMPLE
    i_iter = int(n_total / i_sample)

    if i_contd > 0:
        kernel = NUTS(model=model,
                      target_accept_prob=accept_rate,
                      adapt_step_size=False,
                      adapt_mass_matrix=True,
                      dense_mass=dense_mass,
                      max_tree_depth=(9, 9),
                      forward_mode_differentiation=False,
                      init_strategy=numpyro.infer.init_to_sample)
    else:
        kernel = NUTS(model=model,
                      target_accept_prob=accept_rate,
                      adapt_step_size=True,
                      adapt_mass_matrix=True,
                      dense_mass=dense_mass,
                      max_tree_depth=(9, 9),
                      forward_mode_differentiation=False,
                      init_strategy=numpyro.infer.init_to_sample)
        
    chain_method = 'sequential' if n_chains == 1 else 'parallel'

    if kwargs.get('dry_run', False):
        return dict(model=model,
                    kernel=kernel,
                    obs_data=data_1d_ind,
                    mutable_state_vars=dict(
                        fwd_model=fwd_model,
                        ),
                    )

    # --- Initial MCMC warmup (num_samples=1) ---
    mcmc = MCMC(kernel, num_samples=1, num_warmup=1,
                num_chains=n_chains, thinning=thin,
                chain_method=chain_method, progress_bar=False)
    
    if chain_method == "parallel":
        chain_ids = [i_chain + c for c in range(n_chains)]
        rng_key = make_run_keys(mcmc_seed, chain_ids=chain_ids, batch_id=0, resume_iter=i_contd)
    else:
        rng_key = make_run_key(mcmc_seed, chain_id=i_chain, batch_id=0, resume_iter=i_contd)
    mcmc.warmup(rng_key, obs_data=data_1d_ind, extra_fields=('potential_energy',))
    params = list(mcmc.get_samples().keys())
    params.append('potential_energy')
    map_params = params.copy()
    if collect_ics == 0 and which_ics == 'varied_ics':
        params.remove('gauss_1d')
    logger.info('params = %s', params)
    logger.info('MAP_params = %s', map_params)
    del mcmc

    # --- Main MCMC execution (i_iter batches) ---
    mcmc = MCMC(kernel, num_samples=i_sample, num_warmup=n_warmup,
                num_chains=n_chains, thinning=thin,
                chain_method=chain_method, progress_bar=False)
    
    map_samples = {}          # will hold per-chain MAP snapshots

    save_base = f'{save_path}'
    state_path = lambda cid, it: f'{save_base}_state_chain{cid}_iter{it}.pkl'
    logger.info('save_base = %s', save_base)


    # Load previous samples if restarting (i_contd > 0)
    if i_contd > 0:
        logger.info('Restart mode: loading states at iter=%d', i_contd)
        # Sanity-check previously persisted sample counts when resuming
        chain_ids = [i_chain + c for c in range(n_chains)]
        sanity_check_restart_counts(save_base=f'{save_path}',
                                    chain_ids=chain_ids,
                                    params=params,
                                    i_contd=i_contd,
                                    i_sample=i_sample)
        if n_chains == 1:
            st = load_hmc_state(state_path(i_chain, i_contd))
            mcmc.post_warmup_state = st
        else:
            per = [load_hmc_state(state_path(i_chain + c, i_contd)) for c in range(n_chains)]
            mcmc.post_warmup_state = merge_hmc_states(per)
        tree_block_until_ready(mcmc.post_warmup_state)
        logger.info("Loaded warmup state for restart.")
        for c in range(n_chains):
            cid = i_chain + c
            chain_key = f'chain_{cid}'
            map_samples[chain_key] = {'potential_energy': jnp.inf}  # sentinel
            old_file = f'{save_base}_samples_chain{cid}.npz'
            if os.path.exists(old_file):
                _, old_map, _ = load_samples_npz(old_file)
                if 'potential_energy' in old_map:
                    map_samples[chain_key].update(old_map)
    elif n_warmup <= 1:
        template_file = kwargs.get("template_state_file", None)
        template_chain = int(kwargs.get("template_chain", i_chain))  # chain_id
        template_iter  = int(kwargs.get("template_iter", 0))         # 0=post-warmup
        template_state = None
        if template_file is not None and os.path.exists(template_file):
            logger.info("Loading template state from file: %s", template_file)
            st = load_hmc_state(template_file)
            template_state = split_hmc_state_by_chain(st, getattr(st, "rng_key", jnp.zeros((1,))).shape[0] if hasattr(st, "rng_key") else 1)[0]
        else:
            cand = state_path(template_chain, template_iter)
            if os.path.exists(cand):
                logger.info("Loading template state: %s", cand)
                template_state = load_hmc_state(cand)
            else:
                raise RuntimeError("Template state not found.")
        mcmc.post_warmup_state = broadcast_state_to_chains(
            template_state,
            n_chains,
            base_seed=mcmc_seed,
            resume_iter=i_contd,
            chain_id_start=i_chain,
        )
        tree_block_until_ready(mcmc.post_warmup_state)
        logger.info("Broadcasted template state to %d chains (rng_keys are independent).", n_chains)
    else:
        mcmc_seed += 12345 * i_chain
        logger.info(f'rng_seed = {mcmc_seed}')
        if chain_method == "parallel":
            chain_ids = [i_chain + c for c in range(n_chains)]
            rng_key = make_run_keys(mcmc_seed, chain_ids=chain_ids, batch_id=0, resume_iter=i_contd)
        else:
            rng_key = make_run_key(mcmc_seed, chain_id=i_chain, batch_id=0, resume_iter=i_contd)
        mcmc.warmup(rng_key, obs_data=data_1d_ind, extra_fields=('potential_energy',), collect_warmup=False)
        tree_block_until_ready(mcmc.post_warmup_state)

        def _check(inv_mass_matrix):
            if not dense_mass:
                return False
            block_key = dense_mass[0]
            if n_chains == 1:
                inv_mass_matrix[dense_mass[0]] = inv_mass_matrix[dense_mass[0]][None, :, :]
            return check_dense_mass_matrix(inv_mass_matrix, dense_mass[0], criteria=0.9)[1]  # -> check_fail
        
        if dense_mass:
            inv_mm = copy.deepcopy(mcmc.post_warmup_state.adapt_state.inverse_mass_matrix)
            check_fail = _check(inv_mm)
            warmup_iter = 0
            while check_fail:
                warmup_iter += 1
                if warmup_iter > MAX_WARMUP_ITERS:
                    raise RuntimeError("Reached maximum warmup iterations without passing dense mass matrix check.")
                
                if chain_method == "parallel":
                    chain_ids = [i_chain + c for c in range(n_chains)]
                    rng_key = make_run_keys(mcmc_seed+10*(warmup_iter+1), chain_ids=chain_ids, batch_id=0, resume_iter=i_contd)
                else:
                    rng_key = make_run_key(mcmc_seed+10*(warmup_iter+1), chain_id=i_chain, batch_id=0, resume_iter=i_contd)
                
                mcmc.warmup(rng_key, obs_data=data_1d_ind, extra_fields=('potential_energy',), collect_warmup=False)
                tree_block_until_ready(mcmc.post_warmup_state)
                inv_mm = copy.deepcopy(mcmc.post_warmup_state.adapt_state.inverse_mass_matrix)
                check_fail = _check(inv_mm)
                logger.info('Warmup iteration %d, dense mass check_fail=%s', warmup_iter, check_fail)
        else:
            logger.info("No dense_mass groups; skipping dense mass-matrix check.")
        for c, st in enumerate(split_hmc_state_by_chain(mcmc.post_warmup_state, n_chains)):
            save_hmc_state(st, state_path(i_chain + c, 0))
        logger.info("Warmup finished and state saved.")

    # -----------------------------
    # MCMC loop: run for i_iter batches
    # -----------------------------
    mean_gauss_1d = [None] * n_chains
    mean_count = [0] * n_chains  # running mean count (per chain)

    # Per-chain IDs and in-memory accumulators
    chain_ids = [i_chain + c for c in range(n_chains)]
    combined = {cid: {p: None for p in params} for cid in chain_ids}
    combined_pk = {cid: dict(auto=None, cross=None, resid=None) for cid in chain_ids} if (do_pk and which_ics == 'varied_ics') else None

    # Load previous samples if restarting (i_contd > 0)
    for c, cid in enumerate(chain_ids):
        old_file = f'{save_base}_samples_chain{cid}.npz'
        if i_contd > 0 and os.path.exists(old_file):
            old_samples, _, old_mean = load_samples_npz(old_file)
            for p in params:
                if p in old_samples:
                    combined[cid][p] = np.asarray(old_samples[p])
            if which_ics == 'varied_ics' and old_mean is not None:
                mean_gauss_1d[c] = np.asarray(old_mean)
                # Adjust mean_count to reflect total samples in old file
                mean_count[c] = i_contd * int(i_sample / thin)

        if do_pk and which_ics == 'varied_ics':
            old_pk = f'{save_base}_pk_ic_chain{cid}.npz'
            if i_contd > 0 and os.path.exists(old_pk):
                z = np.load(old_pk)
                combined_pk[cid]['auto']  = np.asarray(z['pk_auto'])  if 'pk_auto'  in z.files else None
                combined_pk[cid]['cross'] = np.asarray(z['pk_cross']) if 'pk_cross' in z.files else None
                combined_pk[cid]['resid'] = np.asarray(z['pk_resid']) if 'pk_resid' in z.files else None
                z.close()

    # Ensure save directory exists
    Path(save_base).parent.mkdir(parents=True, exist_ok=True)

    for i in range(i_iter):
        logger.info("running batch %d ...", i)
        if chain_method == "parallel":
            run_key = make_run_keys(mcmc_seed, chain_ids=chain_ids, batch_id=i, resume_iter=i_contd)
        else:
            run_key = make_run_key(mcmc_seed, chain_id=i_chain, batch_id=i, resume_iter=i_contd)
        
        mcmc.run(run_key,
                 obs_data=data_1d_ind,
                 extra_fields=('potential_energy',))
        tree_block_until_ready(mcmc.last_state)

        dev_samples = mcmc.get_samples(group_by_chain=True)
        dev_samples['potential_energy'] = mcmc.get_extra_fields(group_by_chain=True)['potential_energy']

        # ---- accumulate samples in memory ----
        for c in range(n_chains):
            cid = i_chain + c
            for p in params:
                arr = np.asarray(dev_samples[p][c]).astype(np.float32)  # (batch, ...)
                if combined[cid][p] is None:
                    combined[cid][p] = arr
                else:
                    combined[cid][p] = np.concatenate([combined[cid][p], arr], axis=0)

        # ---- accumulate IC power spectra if needed ----
        if (do_pk and which_ics == 'varied_ics') and ('gauss_1d' in dev_samples):
            for c in range(n_chains):
                cid = i_chain + c
                g_batch = dev_samples['gauss_1d'][c]  # (batch, ng3), on device
                if xres_pk_fn is not None:
                    A, C, R = xres_pk_fn(g_batch)   # device -> host
                    A = np.asarray(A); C = np.asarray(C); R = np.asarray(R)
                    if combined_pk[cid]['auto'] is None:  combined_pk[cid]['auto']  = A
                    else:                                  combined_pk[cid]['auto']  = np.concatenate([combined_pk[cid]['auto'],  A], axis=0)
                    if combined_pk[cid]['cross'] is None: combined_pk[cid]['cross'] = C
                    else:                                  combined_pk[cid]['cross'] = np.concatenate([combined_pk[cid]['cross'], C], axis=0)
                    if combined_pk[cid]['resid'] is None: combined_pk[cid]['resid'] = R
                    else:                                  combined_pk[cid]['resid'] = np.concatenate([combined_pk[cid]['resid'], R], axis=0)
                else:
                    A = np.asarray(auto_pk_fn(g_batch))
                    if combined_pk[cid]['auto'] is None:
                        combined_pk[cid]['auto'] = A
                    else:
                        combined_pk[cid]['auto'] = np.concatenate([combined_pk[cid]['auto'], A], axis=0)

        # ---- update MAP samples if needed ----
        for c in range(n_chains):
            chain_id = i_chain + c
            chain_key = f'chain_{chain_id}'

            # Running mean for ICs
            if which_ics == 'varied_ics':
                ic_samples = dev_samples['gauss_1d'][c]  # (batch, ng3)
                batch_mean = np.asarray(jnp.mean(ic_samples, axis=0))
                bsz = int(ic_samples.shape[0])
                if mean_gauss_1d[c] is None:
                    mean_gauss_1d[c] = batch_mean
                    mean_count[c] = bsz
                else:
                    mean_gauss_1d[c] = (mean_gauss_1d[c] * mean_count[c] + batch_mean * bsz) / (mean_count[c] + bsz)
                    mean_count[c] += bsz

            # Choose the per-batch MAP index by potential energy
            pe = dev_samples['potential_energy'][c]
            idx_min = int(jnp.argmin(pe))
            new_pe = float(np.asarray(pe[idx_min]))

            # Only update if strictly better (or if this is the first iteration after restart)
            if (i_contd == 0) or (new_pe <= float(np.asarray(map_samples[chain_key]['potential_energy']))):
                # Materialize the new MAP snapshot (host-side, float32 to save space)
                new_map = {p: np.asarray(dev_samples[p][c, idx_min]).astype(np.float32) for p in map_params}
                map_samples[chain_key] = new_map

                # Log the fact of update, including batch and chain ids
                logger.info("Updated MAP for %s at batch=%d (global_iter=%d) with potential_energy=%.6g",
                            chain_key, i, i_contd + i + 1, new_pe)

                # Print MAP values (safe and concise):
                # - Skip 'gauss_1d' by default; change blacklist=None if you DO want to preview it.
                _log_map_update(chain_key,
                                new_map,
                                preview_elems=8,
                                blacklist=('gauss_1d',))

        # ---- save samples to file (npz, atomic replace) ----
        for c, cid in enumerate(chain_ids):
            arrays_to_save = {}
            for p in params:
                if combined[cid][p] is None:
                    arrays_to_save[p] = np.empty((0,), dtype=np.float32)
                else:
                    arrays_to_save[p] = combined[cid][p]

            tmp_samples = f'{save_base}_samples_chain{cid}.npz.tmp'
            final_samples = f'{save_base}_samples_chain{cid}.npz'
            save_samples_npz(tmp_samples, arrays_to_save, map_samples[f'chain_{cid}'], mean_gauss_1d[c] if which_ics == 'varied_ics' else None)
            os.replace(tmp_samples, final_samples)
            logger.info('Saved %s', final_samples)

            # ---- save IC power spectra if needed (npz, atomic replace) ----
            if do_pk and which_ics == 'varied_ics' and combined_pk is not None:
                out_pk = {}
                if combined_pk[cid]['auto']  is not None: out_pk['pk_auto']  = combined_pk[cid]['auto'].astype(np.float32)
                if combined_pk[cid]['cross'] is not None: out_pk['pk_cross'] = combined_pk[cid]['cross'].astype(np.float32)
                if combined_pk[cid]['resid'] is not None: out_pk['pk_resid'] = combined_pk[cid]['resid'].astype(np.float32)
                if len(out_pk) > 0:
                    tmp_pk = f'{save_base}_pk_ic_chain{cid}.npz.tmp'
                    final_pk = f'{save_base}_pk_ic_chain{cid}.npz'
                    with open(tmp_pk, 'wb') as f:
                        np.savez_compressed(f, **out_pk)
                    os.replace(tmp_pk, final_pk)
                    logger.info('Saved %s', final_pk)

        # ---- save states to file (pkl, atomic replace) ----
        mcmc.post_warmup_state = mcmc.last_state
        chk_iter = i_contd + i + 1
        chain_states = split_hmc_state_by_chain(mcmc.last_state, n_chains)
        for c, st in enumerate(chain_states):
            tmp  = f"{save_base}_state_chain{i_chain + c}_iter{chk_iter}.pkl.tmp"
            final = state_path(i_chain + c, chk_iter)
            save_hmc_state(st, tmp)
            os.replace(tmp, final)  # atomic
            prev = state_path(i_chain + c, chk_iter - 1)
            if os.path.exists(prev):
                try:
                    os.remove(prev)
                except Exception:
                    pass

        if i == i_iter - 1:
            logger.info("Saved last states for restart at iter=%d", chk_iter)

    # ---- Final MAP logging ----
    for c in range(n_chains):
        cid = i_chain + c
        chain_key = f'chain_{cid}'
        logger.info('MAP samples (final) in chain %d:', cid)
        for param in map_params:
            logger.info('%s: %s', param, map_samples[chain_key][param])
