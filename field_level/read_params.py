#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import shlex
from pathlib import Path

# -------- Helpers (English comments) --------

def _strip_inline_comment(line: str) -> str:
    """Remove inline comments starting with '#' (outside quotes)."""
    # shlex handles quoted strings correctly; we only need to cut at first unquoted '#'
    # Simple approach: split once and keep before '#'
    if "#" in line:
        # But allow quoted '#': use shlex if needed; for now, assume params don't quote '#'
        return line.split("#", 1)[0].strip()
    return line.strip()

def _parse_numeric_value(value_str: str):
    """
    Parse either a scalar or a pair "x y" as float(s).
    Returns float or (float, float).
    """
    parts = value_str.split()
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    return float(parts[0])

def _all_tokens_numeric(value_str: str) -> bool:
    """Return True iff every whitespace-separated token parses as float."""
    parts = value_str.split()
    if not parts:
        return False
    for p in parts:
        try:
            float(p)
        except Exception:
            return False
    return True

def _to_int(value: str):
    return int(float(value))  # allow "128.0" style inputs

def _emit(stderr, key, val):
    print(key, val, file=stderr)

# -------- Main reader --------

def read_params(params_file: str):
    print('params_file =', params_file, file=sys.stderr)

    cosmo_params = {}
    bias_params  = {}
    bias_ties    = {}
    err_params   = {}

    # Predeclare keys (we'll fill them if present).
    data_path = save_path = None
    boxsize = redshift = None
    which_pk = None
    which_ics = None
    collect_ics = None
    ng = ng_L = ng_E = None
    model_name = None
    window_order = 2
    interlace = 1
    which_space = "k_space"
    dense_mass = None
    kmax = None

    # MCMC parameters
    i_chain = 0
    thin = 1
    n_samples = 1000
    n_warmup = 1000
    accept_rate = 0.8
    mcmc_seed = 0
    i_contd = 0
    n_chains = 1

    # IC P(k) measurement options
    pk_nbin = None
    pk_kmin = None
    pk_kmax = None

    true_gauss_3d = None

    ppath = Path(params_file)

    with ppath.open() as f:
        for raw in f:
            line = _strip_inline_comment(raw)
            if not line:
                continue
            # Use shlex to keep quoted values intact
            toks = shlex.split(line, comments=False, posix=True)
            if not toks:
                continue
            key = toks[0]
            value = " ".join(toks[1:]) if len(toks) > 1 else ""

            if key == 'data_path':
                data_path = value
            elif key == 'save_path':
                save_path = value
            elif key == 'boxsize':
                boxsize = float(value)
            elif key == 'redshift':
                redshift = float(value)
            elif key == 'which_pk':
                which_pk = value
            elif key == 'which_ics':
                which_ics = value
            elif key == 'collect_ics':
                collect_ics = _to_int(value)
            elif key == 'ng_L':
                ng_L = _to_int(value)
            elif key == 'ng_E':
                ng_E = _to_int(value)
            elif key == 'ng':
                ng = _to_int(value)
            elif key == 'model_name':
                model_name = value
            elif key == 'window_order':
                window_order = _to_int(value)
            elif key == 'interlace':
                interlace = _to_int(value)
            elif key == 'which_space':
                which_space = value
            elif key == 'dense_mass':
                # Example:
                # dense_mass sigma8 oc hubble log_Perr
                # (Whitespace separated list -> single group/tuple)
                parts = value.split()
                dense_mass = [tuple(parts)] if parts else None
            elif key in ['A', 'sigma8', 'oc', 'hubble', 'ob', 'ns', 'ln1010As']:
                cosmo_params[key] = _parse_numeric_value(value)
            elif key in ['Ab1', 'b1', 'A2b2', 'b2', 'A3bdG2', 'bdG2',
                         'A2bG2', 'bG2', 'A3b3', 'b3', 'A3bG3', 'bG3',
                         'A3bGamma3', 'bGamma3', 'c0', 'c2', 'c4',
                         'Sigma2', 'Sigma2_mu2', 'Sigma2_mu4']:
                # If the value is numeric, treat it as prior/fixed as before.
                # If non-numeric (e.g. "bG2 b1"), treat as a tie: raw -> raw.
                val = value.strip()
                if val and _all_tokens_numeric(val):
                    bias_params[key] = _parse_numeric_value(val)
                else:
                    if val:
                        target = val.split()[0]
                        bias_ties[key] = target
            elif key in ['log_Perr', 'log_Peded', 'Perr_k2', 'log_Perr_k2mu2']:
                err_params[key] = _parse_numeric_value(value)
            elif key == 'kmax':
                kmax = float(value)
            elif key == 'i_chain':
                i_chain = _to_int(value)
            elif key == 'thin':
                thin = _to_int(value)
            elif key == 'n_samples':
                n_samples = _to_int(value)
            elif key == 'n_warmup':
                n_warmup = _to_int(value)
            elif key == 'i_contd':
                i_contd = _to_int(value)
            elif key == 'n_chains':
                n_chains = _to_int(value)
            elif key == 'accept_rate':
                accept_rate = float(value)
            elif key == 'mcmc_seed':
                mcmc_seed = _to_int(value)
            elif key == 'pk_nbin':
                pk_nbin = _to_int(value)
            elif key == 'pk_kmin':
                pk_kmin = float(value)
            elif key == 'pk_kmax':
                pk_kmax = float(value)
            elif key == 'true_gauss_3d':
                true_gauss_3d = value
            # unknown keys are ignored silently

    # Pack compound params
    ics_params = [which_ics, collect_ics]
    mas_params = [window_order, interlace]
    ng_params  = [ng, ng_L, ng_E]
    mcmc_params = [i_chain, n_chains, thin, n_samples, n_warmup, accept_rate, mcmc_seed, i_contd]
    pk_params   = [pk_nbin, pk_kmin, pk_kmax]

    params = {
        'boxsize': boxsize,
        'redshift': redshift,
        'which_pk': which_pk,
        'data_path': data_path,
        'save_path': save_path,
        'ics_params': ics_params,
        'model_name': model_name,
        'ng_params': ng_params,
        'mas_params': mas_params,
        'which_space': which_space,
        'cosmo_params': cosmo_params,
        'bias_params': bias_params,
        'bias_ties': bias_ties,
        'err_params': err_params,
        'kmax': kmax,
        'dense_mass': dense_mass,
        'mcmc_params': mcmc_params,
        'pk_params': pk_params,
        'true_gauss_3d': true_gauss_3d,
    }

    # Echo to stderr for reproducibility
    for k, v in params.items():
        _emit(sys.stderr, k, v)

    return params
