# -*- coding: utf-8 -*-
"""
Tomography tools for nDTomo

@author: Antony Vamvakeros
"""

import numpy as np
from skimage.transform import iradon, radon
from scipy import sparse, ndimage
import astra, time


def radonvol(vol, nproj, scan = 180):
    
    '''
    Calculates the radon transform of a stack of images, 3rd dimension is z/spectral
    '''
    
    theta = np.arange(0, scan, scan/nproj)
    
    s = np.zeros((vol.shape[0], nproj, vol.shape[2]))
    
    for ii in range(s.shape[2]):
        
        s[:,:,ii] = radon(vol[:,:,ii], theta)
    
        print(ii)           
        
    
    print('The dimensions of the sinogram volume are ', s.shape)
        
    return(s)
        
def fbpvol(svol, scan = 180):
    
    '''
    Calculates the reconstructed images of a stack of sinograms using the filtered backprojection algorithm, 3rd dimension is z/spectral
    '''
    
    nt = svol.shape[0]
    nproj = svol.shape[1]
    
    theta = np.arange(0, scan, scan/nproj)
    
    vol = np.zeros((nt, nt, svol.shape[2]))
    
    for ii in range(svol.shape[2]):
        
        vol[:,:,ii] = iradon(svol[:,:,ii], theta, nt, circle = True)
    
        print(ii)           
        
    
    print('The dimensions of the reconstructed volume are ', vol.shape)
        
    return(vol)


def proj_com(projs):
    
    '''
    
    Calculate the centre of mass of projection images (3D volume)
    3rd dimenion is the projection angle
    
    @author: Dorota Matras
    
    '''

    ver = np.zeros((projs.shape[2]))
    
    hor = np.zeros((projs.shape[2]))   
    
    
    for aa in range(projs.shape[2]):
    
        com = ndimage.measurements.center_of_mass(projs[:,:,aa])
    
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
    
    for aa in range(0,projs.shape[2]):
    
    
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



def scalesinos(sinograms):
    
    """
    Method for normalising a sinogram volume (translations x projections x nz/spectral).
    It assumes that the total intensity per projection is constant.
    """  
    
    ss = np.sum(sinograms,axis = 2)
    scint = np.zeros((sinograms.shape[1]))
    # Summed scattering intensity per linescan
    for ii in range(0,sinograms.shape[1]):
        scint[ii] = np.sum(ss[:,ii])
    # Scale factors
    sf = scint/np.max(scint)
    
    # Normalise the sinogram data    
    for jj in range(0,sinograms.shape[1]):
        sinograms[:,jj,:] = sinograms[:,jj,:]/sf[jj] 
    
    return(sinograms)

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
                xxy = np.delete(xxy[I])
                yxy = np.delete(yxy[I])
               
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
    A = sparse.csr_matrix((vals,(rows,cols)),shape=(p*nA,N**2), dtype=np.float)
        
#    A.eliminate_zeros()
#        % Create phantom head as a reshaped vector.
    x = myphantom(N)

#        % Create rhs.
    b = A*x
#    
    return A, b, x#, theta, p, w


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

        

    