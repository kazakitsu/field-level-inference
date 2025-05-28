#!/usr/bin/env python3

import sys

def read_params(params_file):
    print('params_file =', params_file, file=sys.stderr)
    
    cosmo_params = {}
    bias_params  = {}
    err_params   = {}
    
    # Initialize variables; they will be assigned if they are found in the file.
    data_path = None
    save_path = None
    boxsize = None
    redshift = None
    which_pk = None
    which_ics = None
    collect_ics = None
    ng_L = None
    ng_E = None
    ng = None
    model_name = None
    window_order = None
    interlace = None
    which_space = None
    dense_mass = None
    kmax = None
    # MCMC parameters:
    i_chain = None
    thin = None
    n_samples = None
    n_warmup = None
    accept_rate = None
    mcmc_seed = None
    i_contd = None
    n_chains = None

    pk_nbin = None
    pk_kmin = None
    pk_kmax = None

    true_gauss_3d = None

    # Helper function to process lines with one or two numerical values.
    def parse_numeric_value(value_str):
        # Split the value string by whitespace
        parts = value_str.split()
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
        else:
            return float(parts[0])
    
    with open(params_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                continue
            # Split only at the first whitespace so that values with spaces are preserved.
            tokens = line.split(maxsplit=1)
            if len(tokens) != 2:
                continue
            key, value = tokens[0], tokens[1]
            
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
                collect_ics = int(value)
            elif key == 'ng_L':
                ng_L = int(value)
            elif key == 'ng_E':
                ng_E = int(value)
            elif key == 'ng':
                ng = int(value)
            elif key == 'model_name':
                model_name = value
            elif key == 'window_order':
                window_order = int(value)
            elif key == 'interlace':
                interlace = int(value)
            elif key == 'which_space':
                which_space = value
            elif key == 'dense_mass':
                # The line is expected to contain a list of elements, e.g.
                # "dense_mass A oc hubble b1 b2 bG2 c0 log_Perr"
                parts = value.split()
                dense_mass_list = []
                for elem in parts:
                    if elem == 'sigma8':
                        dense_mass_list.append('scaled_sigma8')
                    elif elem == 'oc':
                        dense_mass_list.append('scaled_oc')
                    elif elem == 'hubble':
                        dense_mass_list.append('scaled_hubble')
                    elif elem == 'Peed':
                        dense_mass_list.append('scaled_Peed')
                    else:
                        dense_mass_list.append(elem)
                dense_mass = [tuple(dense_mass_list),]
            elif key in ['A', 'sigma8', 'oc', 'hubble', 'ob', 'ns']:
                cosmo_params[key] = parse_numeric_value(value)
            elif key in ['Ab1', 'b1', 'A2b2', 'b2', 'A3bG2d', 'bG2d',
                         'A2bG2', 'bG2', 'A3b3', 'b3', 'A3bG3', 'bG3',
                         'A3bGamma3', 'bGamma3', 'c0', 'c2', 'c4', 'Sigma2', 'Sigma2_mu2']:
                bias_params[key] = parse_numeric_value(value)
            # For err parameters. For log_Perr and log_Peded, use tuple if two numbers provided.
            elif key in ['log_Perr', 'log_Peded', 'Perr_k2', 'log_Perr_k2mu2',]:
                err_params[key] = parse_numeric_value(value)
            # For other err parameters, read single value.
            elif key in ['fixed_log_Perr', 'fixed_log_Peed', 
                         'fixed_Peded', 'Peed', 'fixed_Peed']:
                err_params[key] = float(value)
            elif key == 'kmax':
                kmax = float(value)
            # MCMC parameters:
            elif key in ['i_chain', 'thin', 'n_samples', 'n_warmup', 'i_contd', 'n_chains']:
                try:
                    val = int(value)
                except ValueError:
                    val = float(value)
                if key == 'i_chain':
                    i_chain = val
                elif key == 'thin':
                    thin = val
                elif key == 'n_samples':
                    n_samples = val
                elif key == 'n_warmup':
                    n_warmup = val
                elif key == 'i_contd':
                    i_contd = val
                elif key == 'n_chains':
                    n_chains = val
            elif key == 'accept_rate':
                accept_rate = float(value)
            elif key == 'mcmc_seed':
                mcmc_seed = int(value)
            elif key in 'pk_nbin':
                pk_nbin = int(value)
            elif key in 'pk_kmin':
                pk_kmin = float(value)
            elif key in 'pk_kmax':
                pk_kmax = float(value)
            elif key == 'true_gauss_3d':
                true_gauss_3d = value

                
    ics_params = [which_ics, collect_ics]
    mas_params = [window_order, interlace]
    ng_params = [ng, ng_L, ng_E]
    mcmc_params = [i_chain, n_chains, thin, n_samples, n_warmup, accept_rate, mcmc_seed, i_contd]

    pk_params = [pk_nbin, pk_kmin, pk_kmax]
    
    params = {}
    params['boxsize'] = boxsize
    params['redshift'] = redshift
    params['which_pk'] = which_pk
    params['data_path'] = data_path
    params['save_path'] = save_path
    params['ics_params'] = ics_params
    params['model_name'] = model_name
    params['ng_params'] = ng_params
    params['mas_params'] = mas_params
    params['which_space'] = which_space
    params['cosmo_params'] = cosmo_params
    params['bias_params'] = bias_params
    params['err_params'] = err_params
    params['kmax'] = kmax
    params['dense_mass'] = dense_mass
    params['mcmc_params'] = mcmc_params
    params['pk_params'] = pk_params
    params['true_gauss_3d'] = true_gauss_3d
    
    for key in params.keys():
        print(key, params[key], file=sys.stderr)
    
    return params
