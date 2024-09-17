#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:03:28 2023

@author: martin
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from OAT import applyDAS, applyUBP, createForwMatdotdet, createForwMatdotdetMIR, createForwMatdotdetLBW
from OAT import build_matrix2, SensorMaskCartCircleArc, build_matrix3
from scipy.interpolate import interp1d
import cv2

def loadingimage(iname,inv):
    cachedir = './'
    imgfile=cachedir+iname+'.tif'
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) # uint8 to float32
    img = img/255 # Normalization
    if inv:
        img = img*-1 + 1
    
    return img 

#filename = 'Mediciones/Mediciones 10-10-2023/med10oct23_3.npz' # OB
#filename = 'Mediciones/Mediciones 20-10-2023/med20oct23_3.npz' # disco 3mm
#filename = 'Mediciones/Mediciones 24-10-2023/med24oct23_2.npz' # disco 3mm
#filename = 'Mediciones/Mediciones 24-10-2023/med24oct23_6.npz' # venas_5p5x5p5mm
filename = 'Mediciones/Mediciones 26-10-2023/med26oct23_1.npz' # venas_12x10mm

#### Mascara para emular muestra acotada en espacio
#x=np.arange(0,nx)
#y=np.arange(0,nx)
#X,Y=np.meshgrid(x,y)
#sig=58; pot=16; M=np.exp(-(((X-nx//2)/sig)**pot+(((Y-nx//2)/sig)**pot)))

data = np.load(filename)
t=data['t']; sinogram=data['sinogram']; angles=data['angles']; Rto=data['Rto'];
Rs=data['Rs']; Fl=data['Fl']; Navg=data['Navg']; sample=data['sample']; vs=data['vs']

Ns, Nt = sinogram.shape
to = t[0]
tf = t[-1]
nx = 128
dx = 115e-6
p = 1*sinogram
dsa = Rs
fmax = 4e6
arco = 360
pn=p/np.max(np.abs(p.ravel()))

Xdas = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p*1)
Xdas = np.reshape(Xdas,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

"""
# Rotate image
pr=np.roll(p,3,0)

Xdas2 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,pr*1)
Xdas2 = np.reshape(Xdas2,(nx,nx))

Xdas2 = np.flip(Xdas2,0)

axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas2,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()
"""

TI = loadingimage('OB',True)
TIm = np.flip(TI,0)

#TIm = loadingimage('disco128',True)
#TIm = loadingimage('venas',True)
#TIm = loadingimage('venas2',True)

axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(TIm,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

posSens = SensorMaskCartCircleArc(Rs, arco, Ns)
Ao = build_matrix2(nx,dx,posSens,vs,to,tf,Nt,False)
#Ao = createForwMatdotdet(Ns,Nt,dx,nx,dsa,arco,vs,to,tf) # without d/dt
PT = Ao@TIm.ravel()
PT = np.reshape(PT,(Ns,Nt))
PT = np.roll(PT,-3,0)
PT = PT*1
PTn=PT/np.max(np.abs(PT.ravel()))

Xdas3 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,PT*1)
Xdas3 = np.reshape(Xdas3,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas3,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

DIR = True
if DIR:
    from transducers import getECO_IR
    from scipy.linalg import convolution_matrix
    source_freq = 2.25e6
    mu = t[Nt//2]#2.04e-6
    sigma = 0.24e-6
    impP = getECO_IR(t,source_freq,mu,sigma)
    ird = impP*-1
    #ird = convolution_matrix(impP,Nt,'same')
    ird = ird.astype(np.float32)


A1 = build_matrix3(nx,dx,posSens,vs,to,tf,Nt,ird)
PT1 = A1@TIm.ravel()
PT1 = np.reshape(PT1,(Ns,Nt))
PT1 = np.roll(PT1,-3,0)
PT1 = PT1*1
PT1n=PT1/np.max(np.abs(PT1.ravel()))

Xdas4 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,PT1*1)
Xdas4 = np.reshape(Xdas4,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas4,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

DIR = True
if DIR:
    from transducers import getECO_IR
    from scipy.linalg import convolution_matrix
    source_freq = 5e6
    mu = t[Nt//2]#2.04e-6
    sigma = 0.24e-6/2 
    impP = getECO_IR(t,source_freq,mu,sigma)
    impP = impP*-1
    ird = convolution_matrix(impP,Nt,'same')
    ird = ird.astype(np.float32)


A2 = build_matrix2(nx,dx,posSens,vs,to,tf,Nt,False,ird)
PT2 = A2@TIm.ravel()
PT2 = np.reshape(PT2,(Ns,Nt))
#PT2 = np.roll(PT2,-3,0)
PT2 = PT2*1
PT2n=PT2/np.max(np.abs(PT2.ravel()))

Xdas5 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,PT2*1)
Xdas5 = np.reshape(Xdas5,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas5,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

#############################################################################
A1 = createForwMatdotdet(Ns,Nt,dx,nx,dsa,arco,vs,to,tf) # with d/dt
PT1 = A1@TIm.ravel()
PT1 = np.reshape(PT1,(Ns,Nt))
PT1 = np.roll(PT1,-3,0)
PT1 = PT1*-1
PT1n=PT1/np.max(np.abs(PT1.ravel()))

Xdas4 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,PT1*1)
Xdas4 = np.reshape(Xdas4,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas4,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

A2 = createForwMatdotdetMIR(Ns,Nt,dx,nx,dsa,arco,vs,to,tf) # with DIR
PT2 = A2@TIm.ravel()
PT2 = np.reshape(PT2,(Ns,Nt))
PT2 = np.roll(PT2,-3,0)
PT2 = PT2*-1
PT2n=PT2/np.max(np.abs(PT2.ravel()))

Xdas5 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,PT2*1)
Xdas5 = np.reshape(Xdas5,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas5,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

A3 = createForwMatdotdetMIR(Ns,Nt,dx,nx,dsa,arco,vs,to,tf) # with DIR and without d/dt
PT3 = A3@TIm.ravel()
PT3 = np.reshape(PT3,(Ns,Nt))
PT3 = np.roll(PT3,-3,0)
PT3 = PT3*1
PT3n=PT3/np.max(np.abs(PT3.ravel()))

Xdas6 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,PT3*1)
Xdas6 = np.reshape(Xdas6,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas6,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

p2=-1*np.gradient(p,axis=1) # Conversion to pressure
p2n=p2/np.max(np.abs(p2.ravel()))
Xdas7 = applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p2*1)
Xdas7 = np.reshape(Xdas7,(nx,nx))
axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdas6,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()

# Sacando parte negativa:
Xdasp = Xdas; Xdasp[Xdasp<0]=0

axis_ticks = [-nx*dx*1e3/2, nx*dx*1e3/2, -nx*dx*1e3/2, nx*dx*1e3/2]
plt.figure(),plt.imshow(Xdasp,extent=axis_ticks, aspect="auto")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title(sample)
plt.colorbar()