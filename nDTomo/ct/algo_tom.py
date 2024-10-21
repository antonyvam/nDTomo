# -*- coding: utf-8 -*-
"""
Tomography tools for nDTomo

@authors: Evangelos Papoutsellis & Antony Vamvakeros
"""
#%%

import algotom.util.utility as util
import algotom.prep.removal as remo

def ring_remover_post_recon_stripe(img, size=300, dim=1, **options):
    (nrow, ncol) = img.shape
    (x_mat, y_mat) = util.rectangular_from_polar(ncol, ncol, ncol, ncol)
    (r_mat, theta_mat) = util.polar_from_rectangular(ncol, ncol, ncol, ncol)
    polar_mat = util.mapping(img, x_mat, y_mat)
    polar_mat = remo.remove_stripe_based_sorting(polar_mat, size=size, dim=dim,  **options)
    mat_rec = util.mapping(polar_mat, r_mat, theta_mat)
    return mat_rec