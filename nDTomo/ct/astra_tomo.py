# -*- coding: utf-8 -*-
"""
Tensorflow functions for tomography

@author: Antony Vamvakeros
"""

import scipy, astra

def Amatrix_astra(ntr, ang):

    '''
    Create A matrix using the astra toolbox
    Might need to delete extra stuff

    Be careful how you define the projection angles
    Example:
    npr = 180
    theta = np.arange(0, 180, 180/npr)
    ang = np.radians(theta)
    '''
    vol_geom = astra.create_vol_geom(ntr, ntr) 
    proj_geom = astra.create_proj_geom('parallel', 1.0, ntr, ang) 
    proj_id = astra.create_projector('line', proj_geom, vol_geom) 
    matrix_id = astra.projector.matrix(proj_id)
    A = astra.matrix.get(matrix_id)
    A = scipy.sparse.csr_matrix.astype(A, dtype = 'float32')

    astra.projector.delete(proj_id)
    astra.data2d.delete(matrix_id)

    return(A)