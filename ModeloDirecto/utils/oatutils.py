import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

##############################################################################
def calcLBP(A,S):
    nx = int(np.sqrt(A.shape[1]))
    Xl = A.T@S.ravel()
    Xl = Xl/np.max(Xl)
    Xl = np.reshape(Xl,(nx,nx))

    return Xl

##############################################################################
def calcsinogram(A,X,Ns,Nt):
    S = A@X.ravel()
    S = S/np.max(S)
    S = np.reshape(S,(Ns,Nt))

    return S

##############################################################################
def plotsquaresinogram(S):
    plt.imshow(S,cmap="viridis",interpolation='nearest',aspect='auto')
    plt.colorbar(),plt.xlabel('$N_t$'),plt.ylabel('$N_s$')

##############################################################################
def addwhitenoise(smax,S):
    # Generación de ruido blanco

    # Adding noise
    # smax = 1e-4 -> 75 dB (ideal)
    # smax = 1e-3 -> 55 dB (very low noise)
    # smax = 5e-3 -> 40 dB (low noise)
    # smax = 1e-2 -> 35 dB (low/moderate noise) 
    # smax = 5e-2 -> 20 dB (moderate noise)
    # smax = 1e-1 -> 15 dB (noisy) 
    # smax = 5e-1 -> 10 dB (almost noise) 

    rm = 0  # white noise mean value
    #smax = 1e-2
    nru = np.random.uniform(0, smax, 1)[0]
    rstd = nru * np.max(np.abs(S))  # noise standard deviation 
            
    noise = np.random.normal(rm, rstd, S.shape)
    noise = noise.astype(np.float32)
    
    Sn = np.zeros(S.shape,dtype=np.float32)
    Sn = S + noise
    
    return Sn

##############################################################################
def loaddata(filename):
    
    data = np.load(filename)
    t=data['t']; sinogram=data['sinogram'];  
    Rs=data['Rs'];  sample=data['sample']; vs=data['vs'];
    Navg=data['Navg']; Rto=data['Rto']; arco=data['arco']; 
    angles=data['angles']; El=data['El'];
    
    return sample, vs, Rs, t, sinogram

##############################################################################
def resamptime(tmed,Smed,Nt):
    tmedn = np.linspace(tmed[0],tmed[-1],Nt).astype(np.float32) # [Nt,]
    Ns = Smed.shape[0]
    Smedn = np.zeros((Ns,Nt)).astype(np.float32) # [Ns,Nt]
    for i1 in range(Ns):
        fp=interp1d(tmed[:],Smed[i1,:])
        aux=fp(tmedn)
        Smedn[i1,:]=aux
    
    return tmedn, Smedn

##############################################################################