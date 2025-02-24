#!/usr/bin/env python3

import sys

def read_params(params_file):
    print('params_file = ', params_file, file=sys.stderr)
    
    cosmo_params = {}
    bias_params  = {}
    err_params   = {}
    
    with open(params_file) as f:
        for line in f:
            if '#' in line:
                continue
            elif 'data_path' in line:
                (idx, data_path) = line.split()
            elif 'save_path' in line:
                (idx, save_path) = line.split()
            elif 'boxsize' in line:
                (idx, value) = line.split()
                boxsize = float(value)
            elif 'redshift' in line:
                (idx, value) = line.split()
                redshift = float(value)
            elif 'which_pk' in line:
                (idx, which_pk) = line.split()
            elif 'which_ics' in line:
                (idx, which_ics) = line.split()
            elif 'collect_ics' in line:
                (idx, value) = line.split()
                collect_ics = int(value)
            elif 'ng_L' in line:
                (idx, value) = line.split()
                ng_L = int(value)
            elif 'ng_E' in line:
                (idx, value) = line.split()
                ng_E = int(value)
            elif 'ng_cut' in line:
                (idx, value) = line.split()
                ng_cut = int(value)
            elif 'ng_e' in line:
                (idx, value) = line.split()
                ng_e = int(value)
            elif 'ng' in line:
                (idx, value) = line.split()
                ng = int(value)
            elif 'model_name' in line:
                (idx, model_name) = line.split()
            elif 'window_order' in line:
                (idx, value) = line.split()
                window_order = int(value)
            elif 'interlace' in line:
                (idx, value) = line.split()
                interlace = int(value)
            elif 'which_space' in line:
                (idx, which_space) = line.split()
            elif 'dense_mass' in line:
                dense_mass_elements = line.split()
                if len(dense_mass_elements) > 1:
                    dense_mass = []
                    for i in range(len(dense_mass_elements)-1):
                        if dense_mass_elements[i+1] == 'sigma8':
                            tmp = 'scaled_sigma8'
                        elif dense_mass_elements[i+1] == 'oc':
                            tmp = 'scaled_oc'
                        elif dense_mass_elements[i+1] == 'hubble':
                            tmp = 'scaled_hubble'
                        elif dense_mass_elements[i+1] == 'Peed':
                            tmp = 'scaled_Peed'
                        else:
                            tmp = dense_mass_elements[i+1]
                        dense_mass.append(tmp)
                    dense_mass = [tuple(dense_mass)]
            elif 'sigma8' in line:
                (idx, value) = line.split()
                cosmo_params[idx] = float(value)
            elif 'oc' in line:
                (idx, value) = line.split()
                cosmo_params[idx] = float(value)
            elif 'hubble' in line:
                (idx, value) = line.split()
                cosmo_params[idx] = float(value)
            elif 'Ab1' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'b1' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'A2b2' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'b2' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'A2bG2' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'bG2' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'A3bGamma3' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'bGamma3' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'A' in line:
                (idx, value) = line.split()
                cosmo_params[idx] = float(value)
            elif 'c0' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'c2' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'c4' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'Sigma2' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'Sigma2_mu2' in line:
                (idx, value) = line.split()
                bias_params[idx] = float(value)
            elif 'fixed_log_Perr' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'log_Perr_k2mu2' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'log_Perr' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'Perr_k2' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'log_Peded' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'fixed_log_Peded' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'Peed' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'fixed_Peed' in line:
                (idx, value) = line.split()
                err_params[idx] = float(value)
            elif 'kmax' in line:
                (idx, value) = line.split()
                kmax = float(value)
            elif 'i_chain' in line:
                (idx, value) = line.split()
                i_chain = int(value)
            elif 'thin' in line:
                (idx, value) = line.split()
                thin = int(value)
            elif 'n_samples' in line:
                (idx, value) = line.split()
                n_samples = int(value)
            elif 'n_warmup' in line:
                (idx, value) = line.split()
                n_warmup = int(value)
            elif 'accept_rate' in line:
                (idx, value) = line.split()
                accept_rate = float(value)
            elif 'mcmc_seed' in line:
                (idx, value) = line.split()
                mcmc_seed = int(value)
            elif 'i_contd' in line:
                (idx, value) = line.split()
                i_contd = int(value)
            elif 'n_chains' in line:
                (idx, value) = line.split()
                n_chains = int(value)

    ics_params = [which_ics, collect_ics]
    mas_params = [window_order, interlace]
    mcmc_params = [i_chain, thin, n_samples, n_chains, n_warmup, accept_rate, mcmc_seed, i_contd]
    
    try:
        ng_params = [ng, ng_L, ng_E, ng_cut, ng_e]
    except NameError:
        ng_params = [ng, ng_L, ng_E]
    
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
    
    for key in params.keys():
        print(key, params[key], file=sys.stderr)

    return params
