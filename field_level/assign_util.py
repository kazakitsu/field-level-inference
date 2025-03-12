#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jax import jit, vmap, custom_vjp
from functools import partial

@partial(jit, static_argnames=('boxsize','window_order','interlace','contd','max_scatter_indices'))
def assign(boxsize, field, weight, pos,
           window_order,
           interlace=0,
           contd=0,
           max_scatter_indices=100_000_000):
    """
    3D version of assign (NGP/CIC/TSC).
      boxsize : float, total size
      field   : shape=(ng, ng, ng)
      weight  : shape=(N,) or scalar
      pos     : shape=(3, N)
      window_order : 1 = NGP, 2 = CIC, 3 = TSC
      interlace    : 0 or 1
      contd        : 0 or 1 (whether to normalize)
      max_scatter_indices : int, if the number of scatter operations exceeds this value, chunk splitting is performed
    Return: Updated field (after scatter)
    """
    # [1] Reshape pos and weight (e.g., from 4D to 2D) is handled here.
    if len(pos.shape) == 4:
        pos = pos.reshape(3, -1)
        weight = weight.reshape(-1)
    return _assign(boxsize, field, weight, pos, window_order, interlace, contd, max_scatter_indices)

# =============================================================================
# [2] 1D Kernel and its Derivative (order = 1: NGP, 2: CIC, 3: TSC)
# =============================================================================
def _kernel_1d(f, shift, order):
    """
    f      : scalar (relative position within the cell, assumed in the range [0, 1))
    shift  : scalar (neighbor offset along that axis; NGP: 0, CIC: 0 or 1, TSC: -1, 0, 1)
    order  : integer corresponding to 1, 2, or 3
    Return : Interpolation weight (scalar)
    """
    if order == 1:
        # NGP: always returns 1 (each particle contributes to only one cell)
        return 1.0
    elif order == 2:
        # CIC: if shift == 0, returns 1 - f; if shift == 1, returns f.
        return jnp.where(shift == 0, 1.0 - f, f)
    elif order == 3:
        # TSC: Standard definition:
        #   W(x) = 0.75 - x^2          if |x| < 0.5
        #          0.5 * (1.5 - |x|)^2   if 0.5 <= |x| < 1.5
        #          0                   otherwise
        # Note: Here, f is the relative position within the cell and shift is one of -1, 0, 1.
        # The following commented-out implementation reflects the standard definition:
        # diff = f - shift  # Here, f is the relative position within the cell, shift is -1, 0, or 1
        # ax = jnp.abs(diff)
        # return jnp.where(ax < 0.5, 0.75 - diff**2,
        #                  jnp.where(ax < 1.5, 0.5*(1.5 - ax)**2, 0.0))
        return jnp.where(
            shift == 0,
            0.75 - f**2,
            0.5 * (f**2 + shift * f + 0.25)
        )
    else:
        raise ValueError(f"Unsupported order: {order}")

def _kernel_1d_grad(f, shift, order):
    """
    Derivative of the 1D kernel with respect to f.
    """
    if order == 1:
        return 0.0
    elif order == 2:
        # Derivative: if shift == 0, d/d f (1 - f) = -1; if shift == 1, d/d f (f) = 1.
        return jnp.where(shift == 0, -1.0, 1.0)
    elif order == 3:
        # For TSC, the derivative (the following commented-out code shows the standard form):
        # diff = f - shift
        # ax = jnp.abs(diff)
        # return jnp.where(ax < 0.5, -2.0 * diff,
        #                  jnp.where(ax < 1.5, -(1.5 - ax) * jnp.sign(diff), 0.0))
        return jnp.where(
            shift == 0,
            -2 * f,
            f + 0.5 * shift
        )
    else:
        raise ValueError(f"Unsupported order: {order}")

# =============================================================================
# [3] Unified Implementation of 3D Kernel Calculation (Forward)
# =============================================================================
def _calc_kernel_3d_unified(fracT, shifts, order):
    """
    fracT : shape (M, 3), relative position within the cell for each particle.
    shifts: shape (n_shifts, 3), neighbor offsets.
    Return: kernel, shape (M, n_shifts), the interpolation weight for each particle and neighbor cell,
            computed as the product over dimensions: ∏₍d₎ _kernel_1d(f[d], shift[d], order)
    """
    def kernel_for_particle(fvec):
        # fvec: shape (3,)
        def kernel_for_neighbor(shift_vec):
            # Compute the value for each axis.
            k = jnp.array([_kernel_1d(fvec[d], shift_vec[d], order) for d in range(3)])
            return jnp.prod(k)
        return vmap(kernel_for_neighbor)(shifts)
    return vmap(kernel_for_particle)(fracT)

# =============================================================================
# [4] Unified Implementation of 3D Kernel Derivative (Backward)
# =============================================================================
def _calc_kernel_3d_grad_unified(fracT, shifts, order):
    """
    fracT : shape (M, 3)
    shifts: shape (n_shifts, 3)
    Return: grad, shape (M, n_shifts, 3)
      For each particle and each neighbor cell, compute the derivative of the interpolation kernel
      with respect to each axis using the chain rule for differentiating a product.
    """
    def grad_for_particle(fvec):
        # fvec: shape (3,)
        # First, compute the kernel and its derivative for each axis.
        def grad_for_neighbor(shift_vec):
            # For each axis d, compute:
            #   k_d = _kernel_1d(fvec[d], shift_vec[d], order)
            #   g_d = _kernel_1d_grad(fvec[d], shift_vec[d], order)
            k = jnp.array([_kernel_1d(fvec[d], shift_vec[d], order) for d in range(3)])
            g = jnp.array([_kernel_1d_grad(fvec[d], shift_vec[d], order) for d in range(3)])
            # The derivative of the kernel with respect to fvec for each axis d is:
            #   g[d] * ∏ (over j ≠ d) k[j]
            grad_vec = jnp.array([g[d] * jnp.prod(jnp.delete(k, d)) for d in range(3)])
            return grad_vec
        return vmap(grad_for_neighbor)(shifts)  # shape (n_shifts, 3)
    return vmap(grad_for_particle)(fracT)

# =============================================================================
# [5] Computation of Interpolation Parameters (Common for Forward/Backward)
# =============================================================================
def _compute_interp_params(pos_mesh, window_order, ng):
    """
    pos_mesh: (3, M)
    Return: imesh, fmesh, shifts
      imesh : (3, M), floor(pos_mesh) (adjusted for TSC)
      fmesh : (3, M), pos_mesh - imesh
      shifts: (n_shifts, 3), neighbor offsets
    """
    if window_order == 1:
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = jnp.zeros_like(pos_mesh)
        shifts = jnp.array([[0, 0, 0]], dtype=jnp.int32)
    elif window_order == 2:
        imesh = jnp.floor(pos_mesh).astype(jnp.int32)
        fmesh = pos_mesh - imesh
        shifts = jnp.stack(jnp.meshgrid(jnp.arange(2), jnp.arange(2), jnp.arange(2),
                                        indexing='ij'), -1).reshape(-1, 3)
    elif window_order == 3:
        imesh = jnp.floor(pos_mesh - 1.5).astype(jnp.int32) + 2
        fmesh = pos_mesh - imesh
        shifts = jnp.stack(jnp.meshgrid(jnp.arange(-1, 2), jnp.arange(-1, 2), jnp.arange(-1, 2),
                                        indexing='ij'), -1).reshape(-1, 3)
    else:
        raise ValueError(f"Unsupported window_order: {window_order}")
    # Apply periodic boundary conditions.
    imesh = jnp.where(imesh < 0, imesh + ng, imesh)
    imesh = jnp.where(imesh >= ng, imesh - ng, imesh)
    return imesh, fmesh, shifts

# =============================================================================
# [6] Forward Scatter (One Chunk): _assign_chunk_impl
# =============================================================================
@partial(jit, static_argnames=('ng', 'window_order'))
def _assign_chunk_impl(field, pos_mesh, weight, ng, window_order):
    """
    pos_mesh: (3, M), weight: (M,)
    """
    imesh, fmesh, shifts = _compute_interp_params(pos_mesh, window_order, ng)
    fracT = fmesh.T  # shape (M, 3)
    kernel = _calc_kernel_3d_unified(fracT, shifts, window_order)  # (M, n_shifts)
    # Indices for each particle's neighbor cells:
    indices = (imesh.T[:, None, :] + shifts[None, :, :])  # (M, n_shifts, 3)
    indices = jnp.where(indices < 0, indices + ng, indices)
    indices = jnp.where(indices >= ng, indices - ng, indices)
    indices = indices.reshape(-1, 3)
    scatter_w = (weight[:, None] * kernel).reshape(-1)
    field = field.at[indices[:, 0], indices[:, 1], indices[:, 2]].add(scatter_w)
    return field

# =============================================================================
# [7] Backward Gather (One Chunk): _assign_chunk_adj
# =============================================================================
def _assign_chunk_adj(field_cot, pos_mesh, weight, ng, cell_size, window_order):
    """
    field_cot: (ng, ng, ng)
    pos_mesh : pos_mesh used in forward, shape = (3, M)
    weight   : (M,)
    Return   : d_pos: (3, M), d_weight: (M,)
    """
    imesh, fmesh, shifts = _compute_interp_params(pos_mesh, window_order, ng)
    fracT = fmesh.T  # shape (M, 3)
    kernel_forward = _calc_kernel_3d_unified(fracT, shifts, window_order)  # (M, n_shifts)
    indices = (imesh.T[:, None, :] + shifts[None, :, :])
    indices = jnp.where(indices < 0, indices + ng, indices)
    indices = jnp.where(indices >= ng, indices - ng, indices)
    indices = indices.reshape(-1, 3)
    gathered = field_cot[indices[:, 0], indices[:, 1], indices[:, 2]]
    gathered = gathered.reshape(-1, shifts.shape[0])  # (M, n_shifts)
    d_weight = jnp.sum(gathered * kernel_forward, axis=1)  # (M,)
    dkernel = _calc_kernel_3d_grad_unified(fracT, shifts, window_order)  # (M, n_shifts, 3)
    dpos_grid = jnp.einsum('ij,i,ijx->ix', gathered, weight, dkernel)  # (M, 3)
    d_pos = (dpos_grid.T) / cell_size  # (3, M)
    return d_pos, d_weight

# =============================================================================
# [8] Chunk Splitting (Common for Forward/Backward)
# =============================================================================
def _chunk_split(num, max_scatter_indices, *arrays):
    if num == 0:
        return None, []
    chunk_size = min(num, max_scatter_indices)
    remainder_size = num % chunk_size
    chunk_num = num // chunk_size
    remainder = None
    if remainder_size > 0:
        remainder_arrays = [x[..., :remainder_size] for x in arrays]
        remainder = tuple(remainder_arrays)
        arrays = [x[..., remainder_size:] for x in arrays]
    chunked_arrays = []
    for x in arrays:
        newshape = x.shape[:-1] + (chunk_num, chunk_size)
        x_resh = x.reshape(newshape)
        chunked_arrays.append(x_resh)
    out = []
    for i in range(chunk_num):
        chunk_i = []
        for arr in chunked_arrays:
            chunk_i.append(arr[..., i, :])
        out.append(tuple(chunk_i))
    return remainder, out

# =============================================================================
# [9] Main Forward Implementation: _assign_impl
# =============================================================================
def _assign_impl(boxsize, field, weight, pos, window_order, interlace=0, contd=0, max_scatter_indices=100_000_000):
    ng = field.shape[0]
    cell_size = boxsize / ng
    N = pos.shape[1]
    pos_mesh = pos / cell_size
    if interlace:
        pos_mesh += 0.5
    if weight.ndim == 0:
        weight_arr = jnp.full((N,), weight)
    else:
        weight_arr = weight
    remainder, chunks = _chunk_split(N, max_scatter_indices, pos_mesh, weight_arr)
    if remainder is not None:
        rpos, rw = remainder
        field = _assign_chunk_impl(field, rpos, rw, ng, window_order)
    for c in chunks:
        cpos, cw = c
        field = _assign_chunk_impl(field, cpos, cw, ng, window_order)
    if not contd:
        field = field / (N / (ng**3))
    return field

# =============================================================================
# [10] Forward Scatter for One Chunk: _assign_chunk_impl (Wrapper Functions)
# =============================================================================
@partial(jit, static_argnames=('ng', 'window_order'))
def _assign_chunk_impl(field, pos_mesh, weight, ng, window_order):
    return _assign_chunk_impl_helper(field, pos_mesh, weight, ng, window_order)

def _assign_chunk_impl_helper(field, pos_mesh, weight, ng, window_order):
    return _assign_chunk_impl_core(field, pos_mesh, weight, ng, window_order)

def _assign_chunk_impl_core(field, pos_mesh, weight, ng, window_order):
    imesh, fmesh, _ = _compute_interp_params(pos_mesh, window_order, ng)
    fracT = fmesh.T  # shape (M, 3)
    kernel = _calc_kernel_3d_unified(fracT, _compute_interp_params(pos_mesh, window_order, ng)[2], window_order)
    indices = (imesh.T[:, None, :] + _compute_interp_params(pos_mesh, window_order, ng)[2])
    indices = jnp.where(indices < 0, indices + ng, indices)
    indices = jnp.where(indices >= ng, indices - ng, indices)
    indices = indices.reshape(-1, 3)
    scatter_w = (weight[:, None] * kernel).reshape(-1)
    field = field.at[indices[:, 0], indices[:, 1], indices[:, 2]].add(scatter_w)
    return field

# =============================================================================
# [11] Backward Gather for One Chunk: _assign_chunk_adj (Re-definition)
# =============================================================================
def _assign_chunk_adj(field_cot, pos_mesh, weight, ng, cell_size, window_order):
    imesh, fmesh, shifts = _compute_interp_params(pos_mesh, window_order, ng)
    fracT = fmesh.T  # shape (M, 3)
    kernel_forward = _calc_kernel_3d_unified(fracT, shifts, window_order)  # (M, n_shifts)
    indices = (imesh.T[:, None, :] + shifts[None, :, :])
    indices = jnp.where(indices < 0, indices + ng, indices)
    indices = jnp.where(indices >= ng, indices - ng, indices)
    indices = indices.reshape(-1, 3)
    gathered = field_cot[indices[:, 0], indices[:, 1], indices[:, 2]]
    gathered = gathered.reshape(-1, shifts.shape[0])  # (M, n_shifts)
    d_weight = jnp.sum(gathered * kernel_forward, axis=1)  # (M,)
    dkernel = _calc_kernel_3d_grad_unified(fracT, shifts, window_order)  # (M, n_shifts, 3)
    dpos_grid = jnp.einsum('ij,i,ijx->ix', gathered, weight, dkernel)  # (M, 3)
    d_pos = (dpos_grid.T) / cell_size  # (3, M)
    return d_pos, d_weight

# =============================================================================
# [12] custom_vjp Forward: _assign_fwd
# =============================================================================
def _assign_fwd(boxsize, field, weight, pos, window_order, interlace, contd, max_scatter_indices):
    out_field = _assign_impl(boxsize, field, weight, pos, window_order, interlace, contd, max_scatter_indices)
    res = (boxsize, field, weight, pos, window_order, interlace, contd, max_scatter_indices)
    return out_field, res

# =============================================================================
# [13] custom_vjp Backward: _assign_bwd
# =============================================================================
def _assign_bwd(boxsize, window_order, interlace, contd, max_scatter_indices, res, field_cot):
    (boxsize, field_init, weight_init, pos_init, window_order, interlace, contd, max_scatter_indices) = res
    ng = field_init.shape[0]
    cell_size = boxsize / ng
    N = pos_init.shape[1]
    scale = 1.0
    if not contd:
        scale = (N / (ng**3))
    d_field = field_cot / scale
    if weight_init.ndim == 0:
        weight_arr = jnp.full((N,), weight_init)
    else:
        weight_arr = weight_init
    d_weight = jnp.zeros_like(weight_arr)
    d_pos = jnp.zeros_like(pos_init)
    pos_mesh = pos_init / cell_size
    if interlace:
        pos_mesh += 0.5
    remainder, chunks = _chunk_split(N, max_scatter_indices, pos_mesh, weight_arr)
    offset = 0
    if remainder is not None:
        rpos, rw = remainder
        dpos_r, dw_r = _assign_chunk_adj(d_field, rpos, rw, ng, cell_size, window_order)
        rsize = rpos.shape[1]
        d_pos = d_pos.at[:, offset:offset + rsize].add(dpos_r)
        d_weight = d_weight.at[offset:offset + rsize].add(dw_r)
        offset += rsize
    for c in chunks:
        cpos, cw = c
        dpos_c, dw_c = _assign_chunk_adj(d_field, cpos, cw, ng, cell_size, window_order)
        csize = cpos.shape[1]
        d_pos = d_pos.at[:, offset:offset + csize].add(dpos_c)
        d_weight = d_weight.at[offset:offset + csize].add(dw_c)
        offset += csize
    if weight_init.ndim == 0:
        d_weight = d_weight.sum()
    return (d_field, d_weight, d_pos)

# =============================================================================
# [14] Setup custom_vjp: _assign
# =============================================================================
@partial(custom_vjp, nondiff_argnums=(0, 4, 5, 6, 7))
def _assign(boxsize, field, weight, pos, window_order, interlace=0, contd=0, max_scatter_indices=100_000_000):
    return _assign_impl(boxsize, field, weight, pos, window_order, interlace, contd, max_scatter_indices)

_assign.defvjp(_assign_fwd, _assign_bwd)

# =============================================================================
# [15] Main Forward Implementation (Re-definition): _assign_impl
# =============================================================================
def _assign_impl(boxsize, field, weight, pos, window_order, interlace=0, contd=0, max_scatter_indices=100_000_000):
    ng = field.shape[0]
    cell_size = boxsize / ng
    N = pos.shape[1]
    pos_mesh = pos / cell_size
    if interlace:
        pos_mesh += 0.5
    if weight.ndim == 0:
        weight_arr = jnp.full((N,), weight)
    else:
        weight_arr = weight
    remainder, chunks = _chunk_split(N, max_scatter_indices, pos_mesh, weight_arr)
    if remainder is not None:
        rpos, rw = remainder
        field = _assign_chunk_impl(field, rpos, rw, ng, window_order)
    for c in chunks:
        cpos, cw = c
        field = _assign_chunk_impl(field, cpos, cw, ng, window_order)
    if not contd:
        field = field / (N / (ng**3))
    return field
