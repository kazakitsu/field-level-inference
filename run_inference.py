#!/usr/bin/env python3
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", type=str, help="Path to the parameter file")
    parser.add_argument("--enable-x64", action="store_true",
                        help="Enable 64-bit mode in JAX", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Optional: enable JAX 64-bit
    import jax
    if args.enable_x64:
        jax.config.update("jax_enable_x64", True)
        print("Enabled 64-bit mode in JAX", file=sys.stderr)

    # Lazy imports (faster CLI feedback)
    import field_level.read_params as read_params
    import field_level.inference as inference

    # Read params from file
    params = read_params.read_params(args.params_file)

    # --- strict validation (no SLURM auto i_chain) ---
    # Require i_chain >= 0 and n_chains >= 1 from the params file.
    i_chain, n_chains = params["mcmc_params"][0], params["mcmc_params"][1]
    if not isinstance(i_chain, int) or i_chain < 0:
        raise ValueError(
            "Invalid i_chain: got {}. You must set a non-negative integer in the params file "
            "(respect-slurm is removed; auto-detection from SLURM_* is no longer supported)."
            .format(i_chain)
        )
    if not isinstance(n_chains, int) or n_chains < 1:
        raise ValueError("Invalid n_chains: got {} (must be >= 1).".format(n_chains))

    # Fire the inference
    inference.field_inference(
        params["boxsize"], params["redshift"], params["which_pk"],
        params["data_path"], params["save_path"],
        params["ics_params"], params["model_name"], params["ng_params"],
        params["mas_params"], params["which_space"],
        params["cosmo_params"], params["bias_params"], params["err_params"],
        params["kmax"], params["dense_mass"], params["mcmc_params"],
        params["pk_params"], params["true_gauss_3d"],
    )
