#!/usr/bin/env python3

import sys
import field_level.inference as inference
import field_level.read_params as read_params

if __name__ == '__main__':
    args = sys.argv
    params_file = args[1]
    
    params_dict = read_params.read_params(params_file)
    
    inference.field_inference(params_dict['boxsize'], params_dict['redshift'], params_dict['which_pk'],
                              params_dict['data_path'], params_dict['save_path'],
                              params_dict['ics_params'], params_dict['model_name'], params_dict['ng_params'], params_dict['mas_params'], params_dict['which_space'],
                              params_dict['cosmo_params'], params_dict['bias_params'], params_dict['err_params'], params_dict['kmax'], 
                              params_dict['dense_mass'], params_dict['mcmc_params'])