import numpy as np
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import math
from scipy import stats
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

##############################################################################
def loadingimage(iname,inv):
    cachedir = './data/'
    imgfile=cachedir+iname+'.tif'
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) # uint8 to float32
    img = img/255 # Normalization
    if inv:
        img = img*-1 + 1
    
    return img 

##############################################################################
def numpynorm(x,vmax,vmin):
    if (np.max(x)-np.min(x))!=0:
        y = (x-np.min(x))/(np.max(x)-np.min(x))
        y = y*(vmax-vmin) + vmin
    else:
        y = x
    return y

##############################################################################
def plotimage(image,suptitle,cols,figsize,titles,fontsize,pad):
    # image: numpy (cols,H,W)
    # cols: int
    # figsize: tuple (int, int)
    # titles: tuple (str,str,...,str) len == cols 
    # fontsize: int
    # pad: float
    # suptitle: insert a main title if suptitle !=''
    
    colormap=plt.cm.viridis
    #colormap=plt.cm.gist_heat
    #colormap=plt.cm.gray
    rows = 1
    fig, ax = plt.subplots(rows,cols,figsize=figsize,dpi=150)
    fig.tight_layout(pad=pad)
    plt.grid(False)
    for i in range(cols):
        #im = ax[0].imshow(image[i,:,:], aspect='equal', interpolation='none', extent=(-tim/2*1e3,tim/2*1e3,-tim/2*1e3,tim/2*1e3),cmap=colormap);
        im = ax[i].imshow(image[i,:,:], aspect='equal', interpolation='none',cmap=colormap);       
        ax[i].set_title(titles[i],fontsize=fontsize);
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="7%", pad=0.15)
        cbar=plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
        for item in ([ax[i].title, ax[i].xaxis.label, ax[i].yaxis.label] +
             ax[i].get_xticklabels() + ax[i].get_yticklabels()):
            item.set_fontsize(fontsize)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(fontsize) 
    if suptitle!='':
        fig.suptitle(suptitle)  
        
##############################################################################
def FoM(Po,Pp):
    
    SSIM=structural_similarity(Po,Pp,data_range=1) 
    PC=stats.pearsonr(Po.ravel(),Pp.ravel())[0]  
    RMSE=math.sqrt(mean_squared_error(Po,Pp))
    PSNR=peak_signal_noise_ratio(Po,Pp)
    
    return SSIM,PC,RMSE,PSNR

##############################################################################
def calcfft(t,y):
    
    dt = t[1]-t[0]
    Nt = t.shape[0]
    Fs = 1/dt
    Nf = 4096
    N0 = int(t[0]/dt)
    #N1 = Nf - N0 - Nt
    #pad = y[0:50]; pad = np.repeat(pad,80)
    yp = np.zeros((Nf,))
    #yp[0:N0] = pad[0:N0]; yp[N0+Nt:]=pad[N0:N0+N1]
    yp[N0:N0+Nt]=y; 
    #plt.plot(yp)
    f = np.linspace(0,Nf,Nf)/Nf*Fs
    f = f[0:Nf//2]
    YF = np.abs(np.fft.fft(yp))
    YF = YF/np.max(YF)
    YF = YF[0:Nf//2]
    
    return f,YF

##############################################################################