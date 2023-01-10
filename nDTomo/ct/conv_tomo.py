# -*- coding: utf-8 -*-
"""
Tomography tools for nDTomo

@author: Antony Vamvakeros
"""
#%%

import numpy as np
from skimage.transform import iradon, radon
from scipy.sparse import csr_matrix
from scipy.ndimage import center_of_mass
from tqdm import tqdm
from scipy.fft import rfft
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

def radonvol(vol, scan = 180, theta=None):
    
    '''
    Calculates the radon transform of a stack of images, 3rd dimension is z/spectral
    '''
    
    if theta is None:
        theta = np.arange(0, scan, scan/vol.shape[0])    
        
    nproj = len(theta)
    
    if len(vol.shape)>2:
    
        s = np.zeros((vol.shape[0], nproj, vol.shape[2]))    
    
        for ii in tqdm(range(s.shape[2])):
            
            s[:,:,ii] = radon(vol[:,:,ii], theta)
                    
    elif len(vol.shape)==2:
        
        s = radon(vol, theta)
        
    
    print('The dimensions of the sinogram volume are ', s.shape)
        
    return(s)
        
def fbpvol(svol, scan = 180, theta=None, nt = None):
    
    '''
    Calculates the reconstructed images of a stack of sinograms using the filtered backprojection algorithm, 3rd dimension is z/spectral
    '''
    if nt is None:
        nt = svol.shape[0]
    nproj = svol.shape[1]
    
    if theta is None:
        theta = np.arange(0, scan, scan/nproj)
    
    if len(svol.shape)>2:
    
        vol = np.zeros((nt, nt, svol.shape[2]))
        
        for ii in tqdm(range(svol.shape[2])):
            
            vol[:,:,ii] = iradon(svol[:,:,ii], theta, nt, circle = True)
                
    elif len(svol.shape)==2:
        
        vol = iradon(svol, theta, nt, circle = True)
    
    print('The dimensions of the reconstructed volume are ', vol.shape)
        
    return(vol)

def airrem(sinograms, ofs = 1):
    
    """
    Method for subtracting the backgroung signal from the sinograms
    Dimensions: translation steps (detector elements), projections, z (spectral)
    """           
    
    di = sinograms.shape
    if len(di)>2:
        for ii in tqdm(range(sinograms.shape[1])):
            air = (np.mean(sinograms[0:ofs,ii,:],axis = 0) + np.mean(sinograms[sinograms.shape[0]-ofs:sinograms.shape[0],ii,:],axis = 0))/2
            sinograms[:,ii,:] = sinograms[:,ii,:] - air

    elif len(di)==2:
        for ii in range(sinograms.shape[1]):
            air = (np.mean(sinograms[0:ofs,:],axis = 0) + np.mean(sinograms[sinograms.shape[0]-ofs:sinograms.shape[0],ii],axis = 0))/2
            sinograms[:,ii] = sinograms[:,ii] - air
        
    return(sinograms)

def scalesinos(sinograms, pbar=False):
    
    """
    Method for normalising a sinogram volume (translations x projections x nz/spectral).
    It assumes that the total intensity per projection is constant.
    Dimensions: translation steps (detector elements), projections, z (spectral)
    """  
    
    di = sinograms.shape
    if len(di)>2:
        ss = np.sum(sinograms,axis = 2)
        scint = np.zeros((sinograms.shape[1]))
        # Summed scattering intensity per linescan
        for ii in range(0,sinograms.shape[1]):
            scint[ii] = np.sum(ss[:,ii])
        # Scale factors
        sf = scint/np.max(scint)
        
        # Normalise the sinogram data    
        if pbar:
            for jj in tqdm(range(sinograms.shape[1])):
                sinograms[:,jj,:] = sinograms[:,jj,:]/sf[jj] 
        else:
            for jj in range(sinograms.shape[1]):
                sinograms[:,jj,:] = sinograms[:,jj,:]/sf[jj]             
            
    elif len(di)==2:

        scint = np.zeros((sinograms.shape[1]))
        # Summed scattering intensity per linescan
        for ii in range(sinograms.shape[1]):
            scint[ii] = np.sum(sinograms[:,ii])
        # Scale factors
        sf = scint/np.max(scint)        
        
        # Normalise the sinogram data    
        if pbar:
            for jj in tqdm(range(sinograms.shape[1])):
                sinograms[:,jj] = sinograms[:,jj]/sf[jj] 
        else:
            for jj in range(sinograms.shape[1]):
                sinograms[:,jj] = sinograms[:,jj]/sf[jj] 
            
    return(sinograms)


def sinocomcor(sinograms):

    """
    Method correcting the sinograms for any motor jitter
    Sinogram can be a 2D or 3D matrix (stack of sinograms)
    Dimensions: translation steps (detector elements), projections, z (spectral)
    """   
    
    di = sinograms.shape
    if len(di)>2:
        ss = np.sum(sinograms, axis = 2)
    else:
        ss = np.copy(sinograms)
            
    com = np.zeros((ss.shape[1],1))
    
    for ii in range(ss.shape[1]):
        
        com[ii,:] = center_of_mass(ss[:,ii])
        
    com = com - com[0]    
    
    sn = np.zeros_like(sinograms)
    
    xold = np.arange(sn.shape[0])


    if len(di)==2:
        
        for ii in tqdm(range(sn.shape[1])):
        
            xnew =  xold + com[ii,:]
                
            sn[:,ii] = np.interp(xnew, xold, sinograms[:,ii])    
        
    elif len(di)>2:
                
        for ll in tqdm(range(sinograms.shape[2])):
            
            for ii in range(sinograms.shape[1]):        
                        
                xnew =  xold + com[ii,:]
                    
                sn[:,ii,ll] = np.interp(xnew, xold, sinograms[:,ii,ll])                    
        
        
    return(sn)

def sinocentering(sinograms, crsr=5, interp=True, scan=180):
            
    """
    Method for centering sinograms by flipping the projection at 180 deg and comparing it with the one at 0 deg
    Sinogram can be a 2D or 3D matrix (stack of sinograms)
    Dimensions: translation steps (detector elements), projections, z (spectral)
    """   
    
    di = sinograms.shape
    if len(di)>2:
        s = np.sum(sinograms, axis = 2)
    else:
        s = np.copy(sinograms)
        
    if scan==360:
        
        s = s[:,0:int(np.round(s.shape[1]/2))]
    
    cr =  np.arange(s.shape[0]/2 - crsr, s.shape[0]/2 + crsr, 0.1)
    
    xold = np.arange(0,s.shape[0])
    
    st = []; ind = [];
    
    print('Calculating the COR')
    
    for kk in tqdm(range(len(cr))):
        
        xnew = cr[kk] + np.arange(-np.ceil(s.shape[0]/2), np.ceil(s.shape[0]/2)-1)
        sn = np.zeros((len(xnew),s.shape[1]), dtype='float32')
        
        
        for ii in range(s.shape[1]):
            
            if interp==True:
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii])
            else:
                
                sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)

        re = sn[::-1,-1]
        st.append((np.std((sn[:,0]-re)))); ind.append(kk)

    m = np.argmin(st)
    print(cr[m])
    
    xnew = cr[m] + np.arange(-np.ceil(s.shape[0]/2), np.ceil(s.shape[0]/2)-1)
    
    print('Applying the COR correction')

    if len(di)>2:
        sn = np.zeros((len(xnew), sinograms.shape[1], sinograms.shape[2]), dtype='float32')  
        for ll in tqdm(range(sinograms.shape[2])):
            for ii in range(sinograms.shape[1]):
                
                if interp==True:
                    sn[:,ii,ll] = np.interp(xnew, xold, sinograms[:,ii,ll])    
                else:
                    sn[:,ii,ll] = np.interp(xnew, xold, sinograms[:,ii,ll], left=0 , right=0) 
            
    elif len(di)==2:
        
        sn = np.zeros((len(xnew),sinograms.shape[1]), dtype='float32')    
        for ii in range(sinograms.shape[1]):
            sn[:,ii] = np.interp(xnew, xold, sinograms[:,ii], left=0 , right=0)
        
    return(sn)

def zigzag(s):
    
    s[0::2,:,:] = s[0::2,::-1,:]
    
    return(s)

def zigzag_flip(im):
    
    im = im[:,0:im.shape[1]-1]
    
    return(im)

def proj_com(projs):
    
    '''
    
    Calculate the centre of mass of projection images (3D volume)
    3rd dimenion is the projection angle
    
    @author: Dorota Matras
    
    '''

    ver = np.zeros((projs.shape[2]))
    
    hor = np.zeros((projs.shape[2]))   
    
    
    for aa in tqdm(range(projs.shape[2])):
    
        com = center_of_mass(projs[:,:,aa])
    
        ver[aa] = com[0]
    
        hor[aa] = com[1]
    
    of_h = hor - hor[0]
    
    of_v = ver - ver[0]
    
    return(of_h, of_v)


def proj_align_vertical(projs, of_v):

    '''
    
    Interpolate projection data vertically to obtain aligned projection data
    3rd dimenion is the projection angle
    
    @author: Dorota Matras
    
    '''
    
    projs_new = np.zeros_like(projs)
         
    # First we align vertically 
    
    for aa in tqdm(range(0,projs.shape[2])):
    
        xold = np.arange(0, projs.shape[0])
    
        xnew =  xold + of_v[aa]
            
        for bb in range(0,projs.shape[1]):
    
            projs_new[:,bb,aa] = np.interp(xnew, xold, projs[:,bb,aa])
    
    return(projs_new)
    
    
def proj_align_horizontal(projs, of_h):

    '''
    
    Interpolate projection data horizontally to obtain aligned projection data
    3rd dimenion is the projection angle
    
    @author: Dorota Matras
    
    '''

    projs_new = np.zeros_like(projs)
         
    # First we align vertically 
    
    for aa in range(0,projs.shape[2]):
    
        xold = np.arange(0, projs.shape[0])
    
        xnew =  xold + of_h[aa]
    
        for bb in range(0,projs.shape[0]):
    
            projs_new[:,bb,aa] = np.interp(xnew, xold, projs[bb,:,aa])
    
    return(projs_new)
     
    
def sino_com_align(vol, crsr = 40, interp0s = False):

    '''
    
    Calculate centre of rotation for each sinogram in stack and align
    Sinograms are stacked along the first dimension
    Returns centered sinograms stacked along the third dimension
    
    @author: Dorota Matras
    
    '''
    
    svol = np.zeros((vol.shape[1], vol.shape[2], vol.shape[0]))
    da = np.zeros((vol.shape[0]))  
    
    for z in range(vol.shape[0]):
    
        s = vol[z,:,:]
        
        cr =  np.arange(s.shape[0]/2 - crsr, s.shape[0]/2 + crsr, 0.1)
    
        xold = np.arange(0,s.shape[0])
    
        st = []; ind = [];
    
        for kk in range(0,len(cr)):
               
            xnew = cr[kk] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2))
    
            sn = np.zeros((len(xnew),s.shape[1]))
    
            for ii in range(0,s.shape[1]):

                if interp0s == True:
    
                    sn[:,ii] = np.interp(xnew, xold, s[:,ii], left=0 , right=0)
                    
                else:
                    
                    sn[:,ii] = np.interp(xnew, xold, s[:,ii])
    
            re = sn[::-1,-1]    
        #        re = sn[:,:1]
    
            st.append((np.std((sn[:,0]-re)))); ind.append(kk)
        
        m = np.argmin(st)
       
        da[z] = cr[m] # you can choose one value – mean from the da stack and rerun this part using cr[m] as a constant value being the median or mean in the da
           
        xnew = cr[m] + np.arange(-np.ceil(s.shape[0]/2),np.ceil(s.shape[0]/2))
    
        for ii in range(0,s.shape[1]):
    
            sn[:,ii] = np.interp(xnew, xold, s[:,ii])  
    
        svol[:,:,z] = sn
        
    return(svol, da)

def create_ramp_filter(s, ang):
    
    N1 = s.shape[1];
    freqs = np.linspace(-1, 1, N1);
    myFilter = np.abs( freqs );
    myFilter = np.tile(myFilter, (len(ang), 1));
    return(myFilter)

def ramp(detector_width):
    
    filter_array = np.zeros(detector_width)
    frequency_spacing = 0.5 / (detector_width / 2.0)
    for i in range(0, filter_array.shape[0]):
        if i <= filter_array.shape[0] / 2.0:
            filter_array[i] = i * frequency_spacing
        elif i > filter_array.shape[0] / 2.0:
            filter_array[i] = 0.5 - (((i - filter_array.shape[0] / 2.0)) * frequency_spacing)
    return filter_array.astype(np.float32)



def rot_center(thetasum):

    '''
    Code taken from: https://github.com/everettvacek/PhaseSymmetry
    If used, please cite: J. Synchrotron Rad. (2022). 29, https://doi.org/10.1107/S160057752101277
    
    The method requires that the object is within the field of view at all projection angles so that the reflection pair arcsine distribution is not obscured, 
    and it also assumes that the rotation axis is aligned to the axis of the 2D detector used to acquire projection images
    
    Calculates the center of rotation of a sinogram.
    Parameters
    ----------
    thetasum : array like
        The 2-D thetasum array (z,theta).
    Returns
    -------
    COR : float
        The center of rotation.    
    '''
    
    T = rfft(thetasum.ravel())
    # Get components of the AC spatial frequency for axis perpendicular to rotation axis.
    imag = T[thetasum.shape[0]].imag
    real = T[thetasum.shape[0]].real
    # Get phase of thetasum and return center of rotation.
    phase = np.arctan2(imag*np.sign(real), real*np.sign(real)) 
    COR = thetasum.shape[-1]/2-phase*thetasum.shape[-1]/(2*np.pi)

    return COR    

def paralleltomo(N, theta, p, w):
    
    '''
    PARALLELTOMO Creates a 2D tomography test problem using parallel beams. 
    
       [A b x theta p w] = paralleltomo(N)
       [A b x theta p w] = paralleltomo(N,theta)
       [A b x theta p w] = paralleltomo(N,theta,p)
       [A b x theta p w] = paralleltomo(N,theta,p,w)
       [A b x theta p w] = paralleltomo(N,theta,p,w,isDisp)
    
    This function creates a 2D tomography test problem with an N-times-N
    domain, using p parallel rays for each angle in the vector theta.
    
    Input: 
       N           Scalar denoting the number of discretization intervals in 
                   each dimesion, such that the domain consists of N^2 cells.
       theta       Vector containing the angles in degrees. Default: theta = 
                   0:1:179.
       p           Number of parallel rays for each angle. Default: p =
                   round(sqrt(2)*N).
       w           Scalar denoting the width from the first ray to the last.
                   Default: w = sqrt(2)*N.
       isDisp      If isDisp is non-zero it specifies the time in seconds 
                   to pause in the display of the rays. If zero (the default), 
                   no display is shown.
    
     Output:
       A           Coefficient matrix with N^2 columns and nA*p rows, 
                   where nA is the number of angles, i.e., length(theta).
       b           Vector containing the rhs of the test problem.
       x           Vector containing the exact solution, with elements
                   between 0 and 1.
       theta       Vector containing the used angles in degrees.
       p           The number of used rays for each angle.
       w           The width between the first and the last ray.
     
     See also: fanbeamtomo, seismictomo.
    
     Jakob Heide Jørgensen, Maria Saxild-Hansen and Per Christian Hansen,
     June 21, 2011, DTU Informatics.
    
     Reference: A. C. Kak and M. Slaney, Principles of Computerized 
     Tomographic Imaging, SIAM, Philadelphia, 2001.   
     
     Original Matlab code from AIR Tools
     Adapted in python by Antony Vamvakeros
     
    '''
    
#    theta = 0:179
#    w = sqrt(2)*N;
#    p = round(sqrt(2)*N);

#    Define the number of angles.
    nA = len(theta);

#    The starting values both the x and the y coordinates. 
    x0 = np.array([np.linspace(-w/2,w/2,p)])
    x0 = np.transpose(x0)
    y0 = np.zeros((p,1));

#    The intersection lines.
    x = np.arange(-N/2,N/2+1)
    y = x

#    Initialize vectors that contains the row numbers, the column numbers and
#    the values for creating the matrix A effiecently.
    rows = np.zeros((2*N*nA*p,))
    cols = np.zeros((2*N*nA*p,))
    vals = np.zeros((2*N*nA*p,))
    idxend = 0


#    thetar = theta*np.pi/180
    thetar = np.deg2rad(theta)

#    Loop over the chosen angles.
    for i in range(0,nA):    
        
        
#        % All the starting points for the current angle.
        x0theta = np.cos(thetar[i])*x0 - np.sin(thetar[i])*y0
        y0theta = np.sin(thetar[i])*x0 + np.cos(thetar[i])*y0
        
#        % The direction vector for all the rays corresponding to the current 
#        % angle.
        a = -np.sin(thetar[i])
        b = np.cos(thetar[i])
        
#        % Loop over the rays.
        for j in range(0,p):
            
#            % Use the parametrisation of line to get the y-coordinates of
#            % intersections with x = k, i.e. x constant.
            tx = (x - x0theta[j])/a
            yx = b*tx + y0theta[j]
            
#            % Use the parametrisation of line to get the x-coordinates of
#            % intersections with y = k, i.e. y constant.
            ty = (y - y0theta[j])/b
            xy = a*ty + x0theta[j]
            
#            % Collect the intersection times and coordinates. 
            t = np.array([tx,ty]);t = np.ndarray.flatten(t)
            xxy = np.array([x,xy]); xxy = np.ndarray.flatten(xxy) 
            yxy = np.array([yx, y]); yxy = np.ndarray.flatten(yxy) 
                        
            
#            % Sort the coordinates according to intersection time.
            
            I = np.argsort(t)
            t = np.sort(t)
            
            xxy = xxy[I]
            yxy = yxy[I]        
            
#            % Skip the points outside the box.
                        
            I = (xxy >= -N/2) & (xxy <= N/2) & (yxy >= -N/2) & (yxy <= N/2)
            xxy = xxy[I]
            yxy = yxy[I]
            
#            % Skip double points.
            I = (np.abs(np.diff(xxy)) <= 1e-10) & (np.abs(np.diff(yxy)) <= 1e-10)
            
#            print(I)
            
            if np.count_nonzero(I)>0:
                try:
                    xxy = np.delete(xxy[I])
                    yxy = np.delete(yxy[I])
                except:
                    pass
                    # xxy = np.delete(xxy[I[:-1]]) # need to fix this
                    # yxy = np.delete(yxy[I[:-1]]) # need to fix this

               
#            % Calculate the length within cell and determines the number of
#            % cells which are hit.
            d = np.sqrt(np.diff(xxy)**2 + np.diff(yxy)**2)
            numvals = d.size
            
            
                   
#            % Store the values inside the box.
            if numvals > 0:
                
#                % If the ray is on the boundary of the box in the top or to the
#                % right the ray does not by definition lie with in a valid cell.
                if ~(((b == 0) & (np.abs(y0theta[j] - N/2) < 1e-15)) | ((a == 0) & (np.abs(x0theta[j] - N/2) < 1e-15))):
                    
#                    % Calculates the midpoints of the line within the cells.
                    xm = 0.5*(xxy[0:-1]+xxy[1:]) + N/2
                    ym = 0.5*(yxy[0:-1]+yxy[1:]) + N/2
    
#                    % Translate the midpoint coordinates to index.
                    col = np.floor(xm)*N + (N - np.floor(ym))
                    
#                    % Create the indices to store the values to vector for
#                    % later creation of A matrix.
                    idxstart = idxend + 0
                    idxend = idxstart + numvals
                    idx = np.arange(idxstart, idxend)
     
#                    % Store row numbers, column numbers and values. 
                    
                    rows[idx] = (i-0.0)*p + j
#                    print(rows[idx])
                    cols[idx] = col - 1
#                    print(cols[idx])
                    vals[idx] = d
#                    print(vals[idx])

#    % Truncate excess zeros.
    rows = rows[0:idxend]
    cols = cols[0:idxend]
    vals = vals[0:idxend]
    
   
#    % Create sparse matrix A from the stored values.
#    A = sparse(rows,cols,vals,p*nA,N^2);
#    A = sparse.coo_matrix((vals,(rows,cols)),shape=(p*nA,N**2))
    A = csr_matrix((vals,(rows,cols)),shape=(p*nA,N**2), dtype=np.float)
        
#    A.eliminate_zeros()
#        % Create phantom head as a reshaped vector.
    # x = myphantom(N)

#        % Create rhs.
    # b = A*x
#    
    return A #, b, x, theta, p, w


def myphantom(N):
    
    '''
    MYPHANTOM creates the modified Shepp-Logan phantom
       X = myphantom(N)
     
     This function create the modifed Shepp-Logan phantom with the
     discretization N x N, and returns it as a vector.
    
     Input:
       N    Scalar denoting the nubmer of discretization intervals in each
            dimesion, such that the phantom head consists of N^2 cells.
     
     Output:
       X    The modified phantom head reshaped as a vector
    
     This head phantom is the same as the Shepp-Logan except the intensities
     are changed to yield higher contrast in the image.
    
     Peter Toft, "The Radon Transform - Theory and Implementation", PhD
     thesis, DTU Informatics, Technical University of Denmark, June 1996.
    
             A    a     b    x0    y0    phi
        ---------------------------------
    
    Original Matlab code from AIR Tools
    Adapted in python by Antony Vamvakeros
    
    '''
    

    e =    np.array([  [1,   .69,   .92,    0,     0,     0],   
                     [-.8,  .6624, .8740,   0,  -.0184,   0],
                     [-.2,  .1100, .3100,  .22,    0,    -18],
                     [-.2,  .1600, .4100, -.22,    0,     18],
                     [.1,  .2100, .2500,   0,    .35,    0],
                     [.1,  .0460, .0460,   0,    .1,     0],
                     [.1,  .0460, .0460,   0,   -.1,     0],
                     [.1,  .0460, .0230, -.08,  -.605,   0 ],
                     [.1,  .0230, .0230,   0,   -.606,   0],
                     [.1,  .0230, .0460,  .06,  -.605,   0   ]])

    xn = (np.arange(0,N)-(N-1)/2)/((N-1)/2)

    Xn = np.tile(xn, (N, 1))
    
    Yn = np.rot90(Xn)
    X = np.zeros((N,N))
         
#     For each ellipse to be added     
    for i in range(0,e.shape[0]):
        
        a2 = e[i,1]**2
        b2 = e[i,2]**2
        x0 = e[i,3]
        y0 = e[i,4]
        phi = e[i,5]*np.pi/180
        A = e[i,0]
        
        x = Xn-x0;
        y = Yn-y0;
        
    
        index = np.where( (x*np.cos(phi) + y*np.sin(phi))**2/a2 + (y*np.cos(phi) - x*np.sin(phi))**2/b2 <= 1)
    
#        % Add the amplitude of the ellipse
        X[index] = X[index] + A;
    
    
#    % Return as vector and ensure nonnegative elements.
    X = np.ndarray.flatten(X) 
    X = np.where(X<0, 0, X)
    
    return X   

def myphantom2(N):
    
    '''
    Original Matlab code from AIR Tools
    Adapted in python by Antony Vamvakeros    
    '''
    
    X = np.zeros((N,N))
    for i in range(0, N):
        for j in range(0,N):
            if (((i-N/2)**2 + (j-N/2)**2) <= (N/2)**2):
                X[i,j]=1
                
            
#    % X(25:35,25:35)=1;
#    % X(45:55,45:55)=0.5;
    X = np.ndarray.flatten(X) 
            
    return X    

def fstomo(N, omegas, rs):

    p = N
    w = N
        
    #    Define the number of angles.
    nA = len(omegas)

    #    The starting values both the x and the y coordinates. 
    x0 = np.array([np.linspace(-w/2,w/2,p)])
    x0 = np.transpose(x0)
    y0 = np.zeros((p,1))

    #    The intersection lines.
    x = np.arange(-N/2,N/2+1)
    y = x

    #    Initialize vectors that contains the row numbers, the column numbers and
    #    the values for creating the matrix A effiecently.
    rows = np.zeros((2*N*nA,))
    cols = np.zeros((2*N*nA,))
    vals = np.zeros((2*N*nA,))
    idxend = 0

    npoints = int(len(rs))

    for point in tqdm(np.arange(npoints)):
                
    #        % All the starting points for the current angle.
        x0theta = np.cos(omegas[point])*x0 - np.sin(omegas[point])*y0
        y0theta = np.sin(omegas[point])*x0 + np.cos(omegas[point])*y0
        
    #        % The direction vector for all the rays corresponding to the current 
    #        % angle.
        a = -np.sin(omegas[point])
        b = np.cos(omegas[point])
        

    #            % Use the parametrisation of line to get the y-coordinates of
    #            % intersections with x = k, i.e. x constant.
        tx = (x - x0theta[int(rs[point])])/a
        yx = b*tx + y0theta[int(rs[point])]
        
    #            % Use the parametrisation of line to get the x-coordinates of
    #            % intersections with y = k, i.e. y constant.
        ty = (y - y0theta[int(rs[point])])/b
        xy = a*ty + x0theta[int(rs[point])]
        
    #            % Collect the intersection times and coordinates. 
        t = np.array([tx,ty]);t = np.ndarray.flatten(t)
        xxy = np.array([x,xy]); xxy = np.ndarray.flatten(xxy) 
        yxy = np.array([yx, y]); yxy = np.ndarray.flatten(yxy) 
                    
        
    #            % Sort the coordinates according to intersection time.
        
        I = np.argsort(t)
        t = np.sort(t)
        
        xxy = xxy[I]
        yxy = yxy[I]        
        
    #            % Skip the points outside the box.
                    
        I = (xxy >= -N/2) & (xxy <= N/2) & (yxy >= -N/2) & (yxy <= N/2)
        xxy = xxy[I]
        yxy = yxy[I]
        
    #            % Skip double points.
        I = (np.abs(np.diff(xxy)) <= 1e-10) & (np.abs(np.diff(yxy)) <= 1e-10)
        
    #            print(I)
        
        if np.count_nonzero(I)>0:
            try:
                xxy = np.delete(xxy[I])
                yxy = np.delete(yxy[I])
            except:
                pass
                # xxy = np.delete(xxy[I[:-1]]) # need to fix this
                # yxy = np.delete(yxy[I[:-1]]) # need to fix this

            
    #            % Calculate the length within cell and determines the number of
    #            % cells which are hit.
        d = np.sqrt(np.diff(xxy)**2 + np.diff(yxy)**2)
        numvals = d.size
        
        
                
    #            % Store the values inside the box.
        if numvals > 0:
            
    #                % If the ray is on the boundary of the box in the top or to the
    #                % right the ray does not by definition lie with in a valid cell.
            if ~(((b == 0) & (np.abs(y0theta[int(rs[point])] - N/2) < 1e-15)) | ((a == 0) & (np.abs(x0theta[int(rs[point])] - N/2) < 1e-15))):
                
    #                    % Calculates the midpoints of the line within the cells.
                xm = 0.5*(xxy[0:-1]+xxy[1:]) + N/2
                ym = 0.5*(yxy[0:-1]+yxy[1:]) + N/2

    #                    % Translate the midpoint coordinates to index.
                col = np.floor(xm)*N + (N - np.floor(ym))
                
    #                    % Create the indices to store the values to vector for
    #                    % later creation of A matrix.
                idxstart = idxend + 0
                idxend = idxstart + numvals
                idx = np.arange(idxstart, idxend)

    #                    % Store row numbers, column numbers and values. 
                
                rows[idx] = point
    #                    print(rows[idx])
                cols[idx] = col - 1
    #                    print(cols[idx])
                vals[idx] = d
    #                    print(vals[idx])

    #    % Truncate excess zeros.
    rows = rows[0:idxend]
    cols = cols[0:idxend]
    vals = vals[0:idxend]


    #    % Create sparse matrix A from the stored values.
    A = csr_matrix((vals,(rows,cols)),shape=(npoints,N**2), dtype=np.float32)

    return(A)


def create_circle(npix_im=512, r0=128):
    
    """
    
    Create a circular mask for a squarred image
    
    """
    im = np.ones((npix_im, npix_im), dtype='float32')
    sz = np.floor(im.shape[0])
    x = np.arange(0,sz)
    x = np.tile(x,(int(sz),1))
    y = np.swapaxes(x,0,1)
    
    xc = np.round(sz/2)
    yc = np.round(sz/2)
    
    r = np.sqrt(((x-xc)**2 + (y-yc)**2))
    
    im = np.where(r>np.floor(sz/2) -(np.floor(sz/2) - r0)+1,0,1)
    im = np.where(r<np.floor(sz/2) -(np.floor(sz/2) - r0), 0, im)

    return(im)
        
def cgls(A, b, K = 25, plot=False):

    
    # Initialization.
    k = np.max(K)
    
    n = A.shape[1]
    
    npix = int(np.sqrt(A.shape[1]))
    
    # Prepare for CG iteration.
    x = np.zeros((n,1))
    
    d = csr_matrix.dot(np.transpose(A),b)
    
    r = b
    normr2 = np.dot(np.transpose(d),d)
    #normr2 = sparse.csr_matrix.dot(np.transpose(d),d)
    
    if plot:
        plt.figure();plt.clf()
    # Iterate.

    for j in range(0,k):
    
    #  Update x and r vectors.
      Ad = csr_matrix.dot(A,d)
      alpha = normr2/(np.dot(np.transpose(Ad),Ad))
    
      x  = x + d*alpha
    
      r  = r - Ad*alpha
      s  = csr_matrix.dot(np.transpose(A),r)
      
    #  Update d vector.
      normr2_new =  np.dot(np.transpose(s),s)
      beta = normr2_new/normr2
      normr2 = normr2_new
      d = s + d*beta
      
    #  Save, if wanted.
      j
    
      xn = x 
      xn = np.reshape(xn,(npix,npix))
      xn = np.where(xn<0, 0, xn)
      xn = xn/np.max(xn)
      xn = np.flipud(np.transpose(xn))
      
      if plot:
          plt.imshow(xn, cmap = 'jet');plt.title(j)
          plt.pause(0.5)
      
    return(xn)
      
    
def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csc_matrix((new_data, new_indices, new_ind_ptr))