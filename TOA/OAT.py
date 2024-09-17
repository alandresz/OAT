# Python Modules:
import numpy as np
#from numpy import matlib
from scipy import sparse
from scipy.sparse import csc_matrix # Column sparse
from scipy.sparse import lil_matrix # Row-based list of lists sparse matrix 
from tqdm import tqdm # progress bar for loops
from scipy.linalg import convolution_matrix
from scipy.interpolate import interp1d
from scipy import signal
import gc
from scipy.fftpack import fft, ifft, fftshift, ifftshift
from scipy.signal import convolve


###############################################################################
def createForwMat(Ns,Nt,dx,nx,dsa,arco,ls,nls,DIS,MDIS,vs,to,tf,AS,SF): # 
    """Creating Forward Model-based Matrix
    """    
    normA = True; thresh = 0;#6; 
    rsnoise = True; tdo = True
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,DIS,MDIS,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp=2*dx/vs)
    #Ao=Ao.astype(np.float32)
    
    return Ao

###############################################################################
def createForwMatdotdet(Ns,Nt,dx,nx,dsa,arco,vs,to,tf): # 
    """Creating Forward Model-based Matrix for point sensors
    """    
    ls = 1e-3; nls = 1; DIS = False; MDIS = 1; AS = True; SF = False
    normA = True; thresh = 0#6; 
    rsnoise = True; tdo = False;
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,DIS,MDIS,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp=2*dx/vs)
    #Ao=Ao.astype(np.float32)
    
    return Ao

###############################################################################
def createForwMatdotdetLBW(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,DIR): # 
    """Creating Forward Model-based Matrix for point sensors with limited bandwidth
    """    
    ls = 1e-3; nls = 1; AS = True; SF = False
    normA = True; thresh = 0;#6;
    rsnoise = True; tdo = True
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    # Detector impulse response (limited bandwidth)
    if DIR:
        MDIR = CreateFilterMatrix(Nt,to,tf)
        MDIR = MDIR.astype(np.float32)
    else:
        MDIR = 1
    Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,DIR,MDIR,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp=2*dx/vs)
    #Ao=Ao.astype(np.float32)
    
    return Ao

###############################################################################
def createForwMatdotdetMIR(Ns,Nt,dx,nx,dsa,arco,vs,to,tf): # 
    """Creating Forward Model-based Matrix for point sensors with measured impulse response
    """    
    ls = 1e-3; nls = 1; AS = True; SF = False
    normA = True; thresh = 0;#6;
    rsnoise = True; tdo = False
    t = np.linspace(to, tf, Nt) # time grid
    ti = t-to
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    # Detector impulse response (limited bandwidth)
    DIR = True
    from transducers import getECO_IR
    source_freq = 2.25e6
    mu = ti[Nt//2]#2.04e-6
    sigma = 0.24e-6  
    impP = getECO_IR(ti,source_freq,mu,sigma)
    impP = impP*-1
    MDIR = convolution_matrix(impP,Nt,'same')
    MDIR = MDIR.astype(np.float32)
    Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,DIR,MDIR,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp=2*dx/vs)
    #Ao=Ao.astype(np.float32)
    
    return Ao

###############################################################################
def applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p): #    
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Pdas = DAS(nx,dx,dsa,posSens,vs,t,p)
    
    return Pdas

###############################################################################
def applyUBP(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p,fmax): #  
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    NFFT = 4096
    #fmax = 4e6
    Pubp = UBP(nx,dx,dsa,posSens,vs,t,p,NFFT,fmax)
    
    return Pubp

###############################################################################
def createrectgrid2D(nx,dx,zo):
    N = int(nx*nx)
    x = np.linspace(-nx*dx/2,nx*dx/2,nx)
    y = np.linspace(-nx*dx/2,nx*dx/2,nx)
    zv = np.ones((nx,nx))*zo
    xv, yv = np.meshgrid(x,y,indexing='xy')
    rj = np.zeros((3,N)) # pixel position [rj]=(3,N))
    rj[0,:] = xv.ravel()
    rj[1,:] = yv.ravel()
    rj[2,:] = zv.ravel()
    rj = rj.astype(np.float32)
    return rj

###############################################################################
def SensorMaskCartCircleArc(circle_radius, circle_arc, num_sensor_points):
    """
    Matrix with the Ns locations (num_sensor_points) of the sensors arranged 
    on a circunference arc (circle_arc) with a radius circle_radius
    """
    th = np.linspace(0, circle_arc * np.pi / 180, num_sensor_points + 1)
    th = th[0:(len(th) - 1)]  # Angles
    posSens = np.array([circle_radius * np.cos(th), circle_radius * np.sin(th), np.zeros((len(th)))])  # position of the center of the sensors
    posSens = posSens.astype(np.float32)
    return posSens # (3,Ns)

###############################################################################
def CreateFilterMatrix(Nt,to,tf):
    
    t = np.linspace(to, tf, Nt) # (Nt,)
    
    # FILTER DESIGN
    Fs = 1/(t[1]-t[0]) # Sampling frequency of the original time grid
    filOrd = 4 # filter order
    
    # Low-pass Filter 
    #flc = 20e6 # cutoff frequency [Hz]
    #bl, al = signal.butter(filOrd, flc, 'low', fs = Fs)
    
    # Band-pass Filter design 
    fbc1 = 0.1e6#1.5e6#1.25e6 # cutoff frequency [Hz]
    fbc2 = 20e6#3.1e6 # 3.8e6 cutoff frequency [Hz]
    bb, ab = signal.butter(filOrd, (fbc1, fbc2), 'bandpass', fs = Fs)
    
    # FILTER IMPULSE RESPONSE
    po=np.zeros(Nt,)
    po[int(Nt/2)]=1
    #impL = signal.filtfilt(bl,al,po)
    impP = signal.filtfilt(bb,ab,po)
    
    # GENERATE FILTER MATRIX
    #FL=convolution_matrix(impL,Nt,'same')
    FP = convolution_matrix(impP,Nt,'same')
    FP = FP.astype(np.float32)
    return FP#,FL

###############################################################################
def build_matrix(nx,dx,Ns,posSens,ls,nls,DIR,MDIR,angsens,SF,vs,tt,normA,thresh,rsnoise,tdo,tlp=0):
    """
    Model-based Matrix by Spatial Impulse Response Approach -> A: (Ns*Nt,N)
    if tdo == False
        VP = A@P0 # where VP: velocity potencial (Ns*Nt,); P0: initial pressure (N,)
    else:
        P = A@P0 # where P: acoustic pressure (Ns*Nt,)

    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    Ns: number of detectors
    posSens: position of the center of the detectors (3,Ns) [m]
    ls: size of the integrating detector [m], length for linear shape and diameter for disc shape
    nls: number of elements of the divided sensor (discretization)
    DIR: if true, measured detector impulse response is used. 
    MDIR: impulse response matrix (Nt, Nt)
    angsens: if True, the surface elements are sensitive to the angle of the incoming wavefront
    SF: if True, the detector shape (disc) is taking into account.
    vs: speed of sound (homogeneous medium) [m/s]
    tt: time samples (Nt,) [s]
    normA: normalize A? True or False
    thresh: threshold the matrix to remove small entries and make it more sparse 10**(-thresh)
    rsnoise: reduce shot noise and add laser pulse duration effect? True or False
    tdo: apply time derivative operator? True or False
    tlp: laser pulse duration [s], by default is set to zero
    
    References:
        [1] G. Paltauf, et al., "Modeling PA imaging with scanning focused detector using
        Monte Carlo simulation of energy deposition", J. Bio. Opt. 23, p. 121607 (2018).
        [2] G. Paltauf, et al., "Iterative reconstruction algorithm for OA imaging",
        J. Acoust. Soc. Am. 112, p. 1536 (2002).
    """    
    # Important constants
    Betta = 207e-6  # Thermal expansion coefficient for water at 20 °C [1/K].
    Calp = 4184     # Specific heat capacity at constant pressure for water at 20 °C [J/K Kg].
    rho = 1000      # Density for water or soft tissue [kg/m^3].
    #h0 = 1e4        # h0=mua*FLa; Energy absorbed in the sample per unit volumen (typical value) [J/m^3].
    #p0=100;         # Initial pressure (typical value) [Pa].
    #phi0 = 5e-9     # phi0=-Dt*p0/rho; Initial velocity potential assuming p0=100Pa and Dt=50ns [m^2/s].

    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  # Total pixels 
    dy = dx  # pixel size in the y direction
    dz = dx  # pixel size in the z direction TODO: 3-D images
    DVol = dx * dy * dz  # element volume    
    rj = createrectgrid2D(nx,dx,0) # Coordinates of the pixels [3,N]
    tprop = (dx+dy+dz)/(3*vs) # Average sound propagation time through a volume element
    
    # SENSOR DISCRETIZATION (for integrating line detectors in the z direction)
    posz = np.zeros((nls,))
    ddot = True
    if nls > 2:
        ddot = False
        posz = np.linspace(-ls / 2, ls / 2, nls) # position of the surface elements (treated as point detectors) of the divided sensor

    # TIME GRID
    dt = tt[1] - tt[0] # time step at which velocity potencials are sampled
    to = int(tt[0] / dt)
    tf = int(tt[len(tt) - 1] / dt) + 1
    sampleTimes = np.arange(to, tf, dtype=int)
    Nt = len(sampleTimes)
    
    # Check if dt <= dx/vs 
    # To avoid loss of information (non-zero matrix), dx and dt should be chosen 
    # such that the average sound propagation time (dx/vs) through a volumen 
    # element (Dvol) is larger than dt, that is, dt <= dx/vs.
    if dt > dx/vs:
        print('ERROR: dt should be smaller than dx/vs!')
        return 0
    
    # SPATIAL IMPULSE RESPONSE (SIR) MATRIX
    print('Creating SIR Matrix...'); # describes the spreading of a delta pulse over the sensor surface
    if rsnoise: # Reduce shot noise induced by the arrival of wave from individual volume elements
        print('with shot noise effect reduction...');
    #Gs = np.zeros((Ns*Nt, N),dtype='float32') # [1/m]
    #Gs = csc_matrix(Gs,dtype='float32')
    Gs = lil_matrix((Ns*Nt, N),dtype='float32')
    currentSens = 0  # Current sensor
    currentTime = 0  # Current time
    for i1 in tqdm(range(Gs.shape[0])):  # Python processing by rows is faster and more efficient
        acum = np.zeros((1, N),dtype='float32') # sum of the velocity potencial detected by each of the surface elements of the divided sensor
        for kk in range(0, nls):  # For each surface element of the divided sensor
            if not(ddot):
                posSens[2,:] = posz[kk]
            # Calculate the distance between DVol and posSensLin
            aux2 = rj - np.reshape(posSens[:,currentSens],(3,1))@np.ones((1,N)) # [3,N]
            R = np.sqrt(aux2[0, 0:] ** 2 + aux2[1, 0:] ** 2 + aux2[2, 0:] ** 2) # [N,]
            # Weight factor: shape factor and angle sensitivity 
            wf = np.ones((1,N),dtype='float32') # [1,N]
            if angsens: # angle sensitivity
                nS = -1*posSens[:,currentSens] # [3,]
                mnS = np.sqrt(nS[0] ** 2 + nS[1] ** 2 + nS[2] ** 2) # norm l2
                nS = nS/mnS # detector surface normal
                nS = np.reshape(nS,(3,1)) @ np.ones((1, N),dtype='float32') # [3,N] 
                wf = np.abs(np.sum(nS*(-1*aux2),axis=0)/R) # cos(tita) = (nS dot (rd-rj))/(|nS|*|rd-rj|) [N,]
                wf = np.reshape(wf, (1, N)) # [1,N]
            if SF: # shape factor
                # Disc shape:
                dsf = 1
                wf = wf * (1 + dsf*np.abs(6.28*posz[kk])/ls) # far from the center, means a larger perimeter (more surface elements for a certain z coordinate)
            R = np.reshape(R, (1, N)) # [1,N]
            # All non-zero elements in a row A belong to DVol whose centers lie within a spherical shell of radius vs*tk and width vs*dt around posSens:
            # delta = 1   si    |t_k - R/vs| < dt/2
            #         0         en otro caso
            
            if rsnoise: # LASER PULSE DURATION EFFECT AND SHOT NOISE REDUCTION?
                # Reduce shot noise induced by the arrival of wave from individual volume elements
                if tprop<tlp:
                    tprop = tlp
                Ti = np.abs(sampleTimes[currentTime] * dt - R / vs)
                #delta = np.exp(-1*((2*Ti/tprop)**2))
                delta = Ti*np.exp(-1*((2*Ti/tprop)**2))
            else:
                #delta = (np.abs(sampleTimes[currentTime] * dt - R / vs)) <= dt / 2 
                #delta = delta * 1  # Bool to int
                delta = Ti*np.exp(-1*((2*Ti/tprop)**2))
            #if tdo:
            #    delta = delta * 1*(R - sampleTimes[currentTime] * dt * vs)
            acum = acum + wf * delta / R  # sum of the velocity potencial detected by each of the surface elements of the divided sensor
        Gs[i1, 0:] = acum
        currentTime = currentTime + 1  
        if np.mod(i1+1, Nt) == 0: # Calculate for other sensor
            currentSens = currentSens + 1
            currentTime = 0
    
    # SYSTEM MATRIZ
    if tdo:
        print('Creating PA Matrix...'); # describes the specific temporal signal from a PA point source
        print('Applying Time Derivative Operator...');
        #Tm=toeplitz(np.arange(0,Nt),np.arange(0,-Nt,-1));
        Tm=np.arange(-np.ceil(Nt/2),np.ceil(Nt/2),dtype=int)
        #Tm=np.arange(-Nt//2+1,Nt//2+1,dtype=int)
        #Tm2=(np.abs(Tm)<=(dx/(vs*dt)))*1; 
        Tm2=(np.abs(Tm)<=(tprop/dt))*1; 
        Tm2=Tm2*Tm
        #tn = np.arange(0,Nt,dtype=int)
        #a=tprop/dt; r=Nt//2; ro = a; f2 = (a**2 - (r-ro-tn)**2)*np.heaviside(a - np.abs(r-ro-tn),0.5)
        #a=tprop/dt; r=Nt//2; ro = -a; f1 = (a**2 - (r-ro-tn)**2)*np.heaviside(a - np.abs(r-ro-tn),0.5)
        #Tm2 = f2-f1
        #po=np.zeros(Nt,)
        #po[int(Nt/2)-2]=1; po[int(Nt/2)+2]=-1;
        #Tm2 = scipy.ndimage.gaussian_filter1d(po, 2)
        Tm2=Tm2*(-vs*dt/(2*dx));
        Gpa=convolution_matrix(Tm2,Nt,'same') # GPa is adimensional
        Gpa=sparse.kron(csc_matrix(np.eye(Ns,dtype='float32')),csc_matrix(Gpa,dtype='float32'))
        del Tm2, Tm
        gc.collect()
        A = Gpa@Gs
        A = (Betta/(4*np.pi*vs**2)*DVol/dt**2)*A;  # A is adimensional
        del Gpa, Gs
        gc.collect()
    else:
        A = Gs
        A = (-Betta / (4 * np.pi * rho * Calp) * DVol / dt) * A  # [m^5/(J*s)]    
    
    if DIR: # Detector Impulse Response
        print('Applying detector impulse response...');
        A = A / abs(A).max()
        A = sparse.kron(np.eye(Ns,dtype='float32'),MDIR)@csc_matrix(A,dtype='float32')
    
    if normA: # System Matrix Normalization
        print('Normalization...')
        A = A / abs(A).max() 
    
    if thresh>0: # Threshold the matrix to remove small entries and make it more sparse
        if normA:
            print('Removing small entries...')
            A = lil_matrix(A)
            A[abs(A)<10**(-thresh)] = 0
            A = lil_matrix(A)
        else:
            print('ERROR: in order to remove small entries, Normalization must be True!')
    
    return A

###############################################################################
def build_matrix2(nx,dx,posSens,vs,to,tf,Nt,Sparse,ird = np.array(0)):
    """
    Model-based Matrix for flat samples -> A: (Ns*Nt,N)
    
        P = A@P0 # where P: acoustic pressure (Ns*Nt,)

    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    to: initial time [s]
    tf: final time [s] 
    Nt: number of time samples
    Sparse: if True, it saves memory but is slower
    ird: impulse response detector matrix (Nt, Nt)
    
    References:
        [1]   
        [2] A. Hauptmann, et al., "Deep learning in PAT: current approaches and
        future directions", J. Bio. Opt. 25, p. 112903 (2020).
    """    

    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  # Total pixels 
    rj = createrectgrid2D(nx,dx,0) # Coordinates of the pixels [3,N]
    rj = rj.astype(np.float32)
    Ns = posSens.shape[1]
    
    # TIME GRID
    cflmax = 1 # Courant–Friedrichs–Lewy condition, cfl = dt*vs/dx <= cfl_{max} = 1
    cfl = 0.3 # recommended value
    dtmax = cflmax*dx/vs # [s]
    Ntmin = int((tf-to)/dtmax)
    dtrec = cfl*dx/vs # [s]
    Ntrec = int((tf-to)/dtrec)
    if Nt < Ntmin:
        print('ERROR: the number of samples time is too low! It is recommended to use',Ntrec,'samples (cfl = 0.3).')
        return 0
    t = np.reshape(np.linspace(to,tf,Nt,dtype=np.float32),(Nt,1)) # (Nt,)
    t = t @ np.ones((1,N),dtype=np.float32) # Time grid matrix (Nt,N)
    
    print('Creating PA Matrix...')
    # Initialise system matrix
    if Sparse:
        A = lil_matrix((Ns*Nt, N),dtype='float32') # (Ns*Nt,nx*nx)
    else:
        A = np.zeros((Ns*Nt,nx**2),dtype=np.float32) # (Ns*Nt,nx*nx)
    
    cont = -1
    for i1 in tqdm(range(0,Ns)):
        cont = cont + 1
        # Distance between the sensor i1 and the pixels of the image region
        R = np.sqrt((posSens[0,i1] - rj[0,:])**2 + (posSens[1,i1] - rj[1,:])**2) # (N,)
        R = np.reshape(R,(1,N)) # (1,N)
        R = np.ones((Nt,1),dtype=np.float32) @ R # (Nt,N)
        
        # Pressure detected by the sensor i1 
        #p = (R - vs*t)/(2*R) * np.heaviside(dx - np.abs(R - vs*t),0.5,dtype = np.float32) # (Nt,N)
        p1 = (dx**2 - (np.abs(R - vs*t))**2)/(2*dx*R) * np.heaviside(dx - np.abs(R - vs*t),0.5,dtype = np.float32) # (Nt,N)
        p2 = (dx**2 - (np.abs(R - vs*(t-4*dx/vs)))**2)/(2*dx*R) * np.heaviside(dx - np.abs(R - vs*(t-4*dx/vs)),0.5,dtype = np.float32) # (Nt,N)
        # 
        A[(Nt*cont):(Nt+Nt*cont),:] = p1-p2
    
    if ird.any():
        print('Applying detector impulse response...')
        A = sparse.kron(np.eye(Ns,dtype='float32'),ird)@csc_matrix(A,dtype='float32')
        #A = lil_matrix(A)
    
    if not(Sparse) and not(ird.any()):
        A = csc_matrix(A,dtype='float32')
    
    del rj,t, R, p1, p2
    gc.collect()    
    
    return A

###############################################################################
def build_matrix3(nx,dx,posSens,vs,to,tf,Nt,ird = np.array(0)):
    """
    Model-based Matrix for flat samples -> A: (Ns*Nt,N)
    
        P = A@P0 # where P: acoustic pressure (Ns*Nt,)

    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    to: initial time [s]
    tf: final time [s] 
    Nt: number of time samples
    ird: impulse response detector matrix (Nt,)
    
    References:
        [1] G. Paltauf, et al., "Characterization of integrating ultrasound
        detectors for PAT", J. App. Phys. 105, p. 102026 (2009).
        [2] A. Hauptmann, et al., "Deep learning in PAT: current approaches and
        future directions", J. Bio. Opt. 25, p. 112903 (2020).
    """    

    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  # Total pixels 
    rj = createrectgrid2D(nx,dx,0) # Coordinates of the pixels [3,N]
    rj = rj.astype(np.float32)
    Ns = posSens.shape[1]
    
    # TIME GRID
    cflmax = 1 # Courant–Friedrichs–Lewy condition, cfl = dt*vs/dx <= cfl_{max} = 1
    cfl = 0.3 # recommended value
    dtmax = cflmax*dx/vs # [s]
    Ntmin = int((tf-to)/dtmax)
    dtrec = cfl*dx/vs # [s]
    Ntrec = int((tf-to)/dtrec)
    if Nt < Ntmin:
        print('ERROR: the number of samples time is too low! It is recommended to use',Ntrec,'samples (cfl = 0.3).')
        return 0
    t = np.linspace(to,tf,Nt,dtype=np.float32) # (Nt,)

    
    print('Creating PA Matrix...')
    # Initialise system matrix
    A = np.zeros((Ns*Nt,N),dtype=np.float32) # (N,Ns*Nt)
    #A = lil_matrix((N,Ns*Nt),dtype='float32') # (Ns*Nt,nx*nx)
    
    applyird = False
    if ird.any():
        print('with detector impulse response...')
        applyird = True
        ird = ird.astype('float32')
    
    with tqdm(total=Ns, leave=True) as pbar:
        cont = -1
        for i1 in range(0,Ns):
            cont = cont + 1
            for i2 in range(0,N):
                # Distance between the sensor i1 and the pixel i2 of the image region
                R = np.sqrt((posSens[0,i1] - rj[0,i2])**2 + (posSens[1,i1] - rj[1,i2])**2) 
                # Pressure detected by the sensor i1 when pixel i2 is the only source
                p1 = (dx**2 - (np.abs(R - vs*t))**2)/(2*dx*R) * np.heaviside(dx - np.abs(R - vs*t),0.5,dtype = np.float32) # (Nt,)
                p2 = (dx**2 - (np.abs(R - vs*(t-4*dx/vs)))**2)/(2*dx*R) * np.heaviside(dx - np.abs(R - vs*(t-4*dx/vs)),0.5,dtype = np.float32) # (Nt,)
                p = p1-p2 # (Nt,)
                if applyird:
                    p = convolve(p, ird, mode='same') # (Nt,)
                A[(Nt*cont):(Nt+Nt*cont),i2] = p
            pbar.update()
    
    print('Applying compressed sparse column matrix...')      
    A = csc_matrix(A,dtype='float32')  
    
    return A

###############################################################################
def DAS(nx,dx,dsa,posSens,vs,t,p):
    """
    Traditional Reconstruction Method "Delay and Sum" for 2-D OAT
    The output P0 is the initial pressure [P0] = (N,) where N is the total pixels
    of the image region.
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    dsa: distance sensor array [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    t: time samples (Nt,) [s]
    p: OA measurements (Ns,Nt) where Ns: number of detectors [Pa]
    
    References:
        [1] X. Ma, et.al. "Multiple Delay and Sum with Enveloping Beamforming 
        Algorithm for Photoacoustic Imaging",IEEE Trans. on Medical Imaging (2019).
    """  
    
    # GET Ns, Nt and N
    Ns=p.shape[0]
    Nt=p.shape[1]
    N = nx**2; # Total pixels 
    
    # 2-D IMAGE REGION GRID
    ny = nx  # nx = ny. Rectangular grid
    N = nx * ny  
    #dy = dx  # pixel size in the y direction 
    
    rj = createrectgrid2D(nx,dx,0) # Coordinates of the pixels [3,N]
    rj = rj[0:2,:] # pixel position [rj]=(2,N)
    rj=np.transpose(rj) # (N,2)
    Rj=np.reshape(rj,(1,N*2)) # (1,2*N)
    Rj=np.repeat(Rj,Ns,axis=0) # (Ns,2*N)
    Rj=np.reshape(Rj,(Ns*N*2,1)) # (2*N*Ns,1)
        
    # DETECTOR POSITIONS
    rs=posSens[0:2,:] # (2,Ns)
    rs=np.transpose(rs) # (Ns,2)
    Rs=np.repeat(rs,N,axis=0) # (N*Ns,2)
    Rs=np.reshape(Rs,(Ns*N*2,1)) # (2*N*Ns,1)
 
    # TIME GRID
    Tau=(Rs-Rj)/vs
    Tau=np.reshape(Tau,(Ns*N,2))
    Tau=np.linalg.norm(Tau,ord=2,axis=1) # Get norm 2 by row
    Tau=np.reshape(Tau,(Ns,N))
    
    # OBTAIN 2-D IMAGE
    P0=np.zeros((N,))
    t = np.reshape(t,(1,Nt))
    #for i in tqdm(range(0,Ns)):
    for i in range(0,Ns):
        fp=interp1d(t[0,:],p[i,:]) #interpolación lineal de la mediciones para los tiempo GRILLA
        aux=fp(Tau[i,:])
        P0=P0+aux
    
    #P0=np.reshape(P0,(nx,nx)) 
    return P0 

###############################################################################
def UBP(nx,dx,dsa,posSens,vs,t,pmed,NFFT,fmax):
    """
    Traditional Reconstruction Method "Universal Back-Projection" for 2-D OAT
    It is assumed detector are on a circunference surrounding the image region,
    all in the XY plane. It is assumed that the detectors are in the same plane
    as the image and they are distributed in a circunference of radius dsa.     
    
    nx: number of pixels in the x direction for a 2-D image region
    dx: pixel size  in the x direction [m]
    dsa: distance sensor array [m]
    posSens: position of the center of the detectors (3,Ns) [m]
    vs: speed of sound (homogeneous medium) [m/s]
    t: time samples (Nt,) [s]
    pmed: OA measurements (Ns,Nt) where Ns: number of detectors [Pa]
    NFFT = size of the vector used for fft.
    fmax = maximum frequency of the ramp filter [Hz]    
    
    Reference:
        [1] M. Xu, et al., "Universal back-projection #algorithm for 
        photoacoustic computed tomography", Phys. Rev. E 71, p. 016706 (2005).
    """    
    
    # GET Ns, Nt and N
    Ns=pmed.shape[0]
    Nt=pmed.shape[1]
    
    # 2-D IMAGE REGION GRID
    rj = createrectgrid2D(nx,dx,0) # Coordinates of the pixels [3,N]
        
    # DETECTOR POSITIONS
    rs = posSens[0:2,:] # (2,Ns)
    
    # Adding zeros to the measured signals
    #NFFT = 4096   
    ceros = np.zeros((NFFT-Nt,))
    
    # BACKPROJECTION
    fs = 1/(t[1]-t[0])
    fv = np.arange(NFFT)/NFFT*fs # [Hz]
    fv = fv-fs/2
    k = 2*np.pi*fv/vs #wave vector
    #fmax = 20e6 # [Hz]
    kmax = 2*np.pi*fmax/vs
       
    Hw = np.abs(k) # [NFFT,]
    Hw = np.where(Hw > kmax, Hw*0, Hw*1) # [NFFT,]
    #Hw = Hw*np.hamming(NFFT)

    #ds = 1e-3*20e-3  #active area of sensor (assuming the size of a line detector)
    ds = np.pi*(13e-3/2)**2  #active area of sensor (assuming the size of the detector V306-SU)
    pf = np.empty([len(k)])
    pnum = 0
    pden = 0
    
    cont = -1
    for i1 in tqdm(range(0,Ns)):
        cont = cont + 1
        
        X2 = (rj[0,:]-rs[0,cont])**2
        Y2 = (rj[1,:]-rs[1,cont])**2
        dist = np.sqrt(X2+Y2)  #distance from detector to each pixel in imaging field
        dist2time = dist/vs 
        
        p = pmed[cont,:]
        pf = fftshift(fft(np.concatenate((p,ceros)))/NFFT)
        pf = -1j * Hw * pf #apply ramp filter
        pf = ifft(ifftshift(pf))
        
        #b = 2*p - 2*t*vs*pf[0:len(p)]
        b = -2*t*vs*pf[0:len(p)]

        fp=interp1d(t,b) 
        b1 = fp(dist2time)
        # Calculation dOmega assuming detectors on a circunference surrounding the sample
        rsni = np.sqrt(rs[0,cont]**2+rs[1,cont]**2)
        etta = -1*(rs[0,cont]*(rj[0,:]-rs[0,cont]) + rs[1,cont]*(rj[1,:]-rs[1,cont]))/rsni
        omega = (ds/dist**2)*etta/dist
        pnum = pnum + omega*b1
        pden = pden + omega

    pg = pnum/pden
    pgmax = pg[np.nonzero(np.abs(pg) == np.amax(abs(pg)))]   #np.amax(complex array) will return the element with maximum real value, not maximum absolute value as in Matlab
    pfnorm = np.real(pg/pgmax)
    return pfnorm