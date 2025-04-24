#!/usr/bin/env python3

import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", type=str, help="Path to the parameter file")
    parser.add_argument("--enable-x64", action="store_true", help="Enable 64-bit mode in JAX", default=False)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    params_file = args.params_file
    enable_x64 = args.enable_x64

    import jax
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    import field_level.read_params as read_params
    import field_level.inference as inference
    
    params_dict = read_params.read_params(params_file)
    
    inference.field_inference(params_dict['boxsize'], params_dict['redshift'], params_dict['which_pk'],
                              params_dict['data_path'], params_dict['save_path'],
                              params_dict['ics_params'], params_dict['model_name'], params_dict['ng_params'], params_dict['mas_params'], params_dict['which_space'],
                              params_dict['cosmo_params'], params_dict['bias_params'], params_dict['err_params'], params_dict['kmax'], 
                              params_dict['dense_mass'], params_dict['mcmc_params'],
                              params_dict['pk_params'], params_dict['true_gauss_3d'],)