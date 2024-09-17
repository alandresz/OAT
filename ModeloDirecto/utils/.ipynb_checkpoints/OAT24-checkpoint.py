# Python Modules:
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix # Column sparse
from scipy.sparse import lil_matrix # Row-based list of lists sparse matrix 
from tqdm import tqdm # progress bar for loops
from scipy.linalg import convolution_matrix
from scipy.interpolate import interp1d
from scipy import signal
import gc

###############################################################################
def createForwMat(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,nhie,opc): # 
    """Creating Forward Model-based Matrix 
    opc = 1 -> ideal point sensors
    opc = 2 -> point sensor with limited bandwidth and angular sensitivity
    opc = 3 -> point sensor with measured impulse response and angular sensitivity
    opc = 4 -> line detector Z with angular sensitivity
    opc = 5 -> line detector XY with measured impulse response and angular sensitivity
    """    
    
    # Model-based matrix parameters
    normA = True; thresh = 0; rsnoise = True; tdo = True; tlp=2*dx/vs
    
    # Time grid
    t = np.linspace(to, tf, Nt) 
    t = t.astype(np.float32)
    
    # Spatial distribution of the sensors around the sample
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) 
    
    # Sensor parameters
    Sdot = True
    if opc == 1:
        print('Sensor type: ideal point detector')
        ls = 1e-3; nls = 1; MDIR = np.array([]); AS = False; SF = False;
    elif opc == 2:
        print('Sensor type: point detector with limited BW and angular sensitivity')
        ls = 1e-3; nls = 1; AS = True; SF = False;
        MDIR = CreateFilterMatrix(Nt,to,tf); MDIR = MDIR.astype(np.float32)
    elif opc == 3:
        print('Sensor type: point detector with measured impulse response and angular sensitivity')
        ls = 1e-3; nls = 1; AS = True; SF = False;
        from transducers import getECO_IR
        ti = t-to
        source_freq = 2.25e6
        mu = ti[Nt//2]#2.04e-6
        sigma = 0.24e-6  
        impP = getECO_IR(ti,source_freq,mu,sigma)
        impP = impP#*-1
        MDIR = convolution_matrix(impP,Nt,'same'); MDIR = MDIR.astype(np.float32)
    elif opc == 4:
        print('Sensor type: line detector Z with angular sensitivity')
        ls = 24e-3; nls = int(ls/dx); MDIR = np.array([]); AS = True; SF = False
    elif opc == 5:
        print('Sensor type: line detector XY with measured impulse response and angular sensitivity')
        ls = 13e-3; nls = int(ls/dx); AS = True; SF = False
        from transducers import getECO_IR
        ti = t-to
        source_freq = 2.25e6
        mu = ti[Nt//2]#2.04e-6
        sigma = 0.24e-6  
        impP = getECO_IR(ti,source_freq,mu,sigma)
        impP = impP#*-1
        MDIR = convolution_matrix(impP,Nt,'same'); MDIR = MDIR.astype(np.float32)
        Ao = build_matrix2D(nx,dx,Ns,posSens,ls,nls,MDIR,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp,nhie)
        Sdot = False
    else:
        print('Default case: ideal point detector')
        ls = 1e-3; nls = 1; MDIR = np.array([]); AS = False; SF = False;
    
    # Calling the matrix function
    if Sdot:
        Ao = build_matrix(nx,dx,Ns,posSens,ls,nls,MDIR,AS,SF,vs,t,normA,thresh,rsnoise,tdo,tlp,nhie)
    
    Ao=Ao.astype(np.float32)
    
    return Ao
   

###############################################################################
def applyDAS(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,p): #    
    t = np.linspace(to, tf, Nt) # time grid
    posSens = SensorMaskCartCircleArc(dsa,arco,Ns) # position of the center of the detectors (3,Ns) [m]
    Pdas = DAS(nx,dx,dsa,posSens,vs,t,p)
    
    return Pdas

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
def spotlaser(rj,z):
    """
    Non homogeneous illumnation effect
    """
    
    # Spot characteristics
    w0 = 0.7e-3; f = -30e-3; l0 = 532e-9; n = 1;
    z0 = np.pi*n*w0**2/l0
    w = np.sqrt(z0*l0/np.pi*((1-z/f)**2+(z/z0)**2));
    # 
    Ilaser = np.exp(-(rj[0,:]**2 + rj[1,:]**2)/2/w**2)
    Ilaser = Ilaser.astype(np.float32)
    
    return Ilaser

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
    fbc1 = 1.5e6#1.25e6 # cutoff frequency [Hz]
    fbc2 = 3.1e6 # 3.8e6 cutoff frequency [Hz]
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
def build_matrix(nx,dx,Ns,posSens,ls,nls,MDIR,angsens,SF,vs,tt,normA,thresh,rsnoise,tdo,tlp,nhie):
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
    MDIR: impulse response matrix (Nt, Nt), if MDIR = np.array([])---> the impulse response is disable
    angsens: if True, the surface elements are sensitive to the angle of the incoming wavefront
    SF: if True, the detector shape (disc) is taking into account.
    vs: speed of sound (homogeneous medium) [m/s]
    tt: time samples (Nt,) [s]
    normA: normalize A? True or False
    thresh: threshold the matrix to remove small entries and make it more sparse 10**(-thresh)
    rsnoise: reduce shot noise and add laser pulse duration effect? True or False
    tdo: apply time derivative operator? True or False
    tlp: laser pulse duration [s], by default is set to zero
    nhie: Non homogeneous illumnation effect? 
    
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
    
    # ILLUMINATION EFFECT
    if nhie > 0:
        print('with illumination effect...')
        Ilaser = spotlaser(rj,nhie) # [1,N]
    else:
        Ilaser = 1        
    
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
    
    cflmax = 1 # Courant–Friedrichs–Lewy condition, cfl = dt*vs/dx <= cfl_{max} = 1
    cfl = 0.3 # recommended value
    dtmax = cflmax*dx/vs # [s]
    Ntmin = int((tt[1]-tt[0])/dtmax)
    dtrec = cfl*dx/vs # [s]
    Ntrec = int((tt[1]-tt[0])/dtrec)
    if Nt < Ntmin:
        print('ERROR: the number of samples time is too low! It is recommended to use',Ntrec,'samples (cfl = 0.3).')
        return 0
    
    # SPATIAL IMPULSE RESPONSE (SIR) MATRIX
    print('Creating SIR Matrix...'); # describes the spreading of a delta pulse over the sensor surface
    if rsnoise: # Reduce shot noise induced by the arrival of wave from individual volume elements
        print('with shot noise effect reduction...');
    if angsens: # angle sensitivity
        print('with angle sensitivity...')
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
                dsf = 1/2
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
                delta = np.exp(-1*((2*Ti/tprop)**2))
            else:
                delta = (np.abs(sampleTimes[currentTime] * dt - R / vs)) <= dt / 2 
                delta = delta * 1  # Bool to int
            #if tdo:
            #    delta = delta * 1*(R - sampleTimes[currentTime] * dt * vs)
            acum = acum + wf * Ilaser * delta / R  # sum of the velocity potencial detected by each of the surface elements of the divided sensor
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
        Tm2=(np.abs(Tm)<=(dx/(vs*dt)))*1; 
        Tm2=Tm2*Tm
        Tm2=Tm2*(-vs*dt/(2*dx));
        Gpa=convolution_matrix(Tm2,Nt,'same') # GPa is adimensional
        Gpa=sparse.kron(csc_matrix(np.eye(Ns,dtype='float32')),csc_matrix(Gpa,dtype='float32'))
        del Tm, Tm2
        gc.collect()
        A = Gpa@Gs
        A = (Betta/(4*np.pi*vs**2)*DVol/dt**2)*A;  # A is adimensional
        del Gpa, Gs
        gc.collect()
    else:
        A = Gs
        A = (-Betta / (4 * np.pi * rho * Calp) * DVol / dt) * A  # [m^5/(J*s)]    
    
    if MDIR.size != 0: # Detector Impulse Response Matrix
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
def build_matrix2D(nx,dx,Ns,posSens,ls,nls,MDIR,angsens,SF,vs,tt,normA,thresh,rsnoise,tdo,tlp,nhie):
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
    ls: size of the integrating detector [m], diameter for disc shape
    nls: number of elements of the divided sensor (discretization)
    MDIR: impulse response matrix (Nt, Nt) if MDIR = np.array([]) ---> the impulse response is disable
    angsens: if True, the surface elements are sensitive to the angle of the incoming wavefront
    SF: if True, the detector shape (disc) is taking into account.
    vs: speed of sound (homogeneous medium) [m/s]
    tt: time samples (Nt,) [s]
    normA: normalize A? True or False
    thresh: threshold the matrix to remove small entries and make it more sparse 10**(-thresh)
    rsnoise: reduce shot noise and add laser pulse duration effect? True or False
    tdo: apply time derivative operator? True or False
    tlp: laser pulse duration [s]
    nhie: Non homogeneous illumnation effect? if nihe == 0 --> is disable
    
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
    DVol = dx**3  # element volume    
    rj = createrectgrid2D(nx,dx,0) # Coordinates of the pixels [3,N]
    rj = rj[0:2,:] # [2,N]
    tprop = (3*dx)/(3*vs) # Average sound propagation time through a volume element
    
    # ILLUMINATION EFFECT
    if nhie > 0:
        print('with illumination effect...')
        Ilaser = spotlaser(rj,nhie) # [1,N]
    else:
        Ilaser = 1   
    
    # SENSOR DISCRETIZATION (for integrating area detectors in the xy directions)
    posz = np.zeros((nls,))
    if nls > 2:
        posz = np.linspace(-ls / 2, ls / 2, nls) # position of the surface elements (treated as point detectors) of the divided sensor

    # TIME GRID
    dt = tt[1] - tt[0] # time step at which velocity potencials are sampled
    to = int(tt[0] / dt)
    tf = int(tt[len(tt) - 1] / dt) + 1
    sampleTimes = np.arange(to, tf, dtype=int)
    Nt = len(sampleTimes)
    
    cflmax = 1 # Courant–Friedrichs–Lewy condition, cfl = dt*vs/dx <= cfl_{max} = 1
    cfl = 0.3 # recommended value
    dtmax = cflmax*dx/vs # [s]
    Ntmin = int((tt[1]-tt[0])/dtmax)
    dtrec = cfl*dx/vs # [s]
    Ntrec = int((tt[1]-tt[0])/dtrec)
    if Nt < Ntmin:
        print('ERROR: the number of samples time is too low! It is recommended to use',Ntrec,'samples (cfl = 0.3).')
        return 0
    
    # SPATIAL IMPULSE RESPONSE (SIR) MATRIX
    print('Creating SIR Matrix...'); # describes the spreading of a delta pulse over the sensor surface
    if rsnoise: # Reduce shot noise induced by the arrival of wave from individual volume elements
        print('with shot noise effect reduction...');
    if angsens: # angle sensitivity
        print('with angle sensitivity...')
    Gs = lil_matrix((Ns*Nt, N),dtype='float32')
    currentSens = 0  # Current sensor
    currentTime = 0  # Current time
    for i1 in tqdm(range(Gs.shape[0])):  # Python processing by rows is faster and more efficient
        acum = np.zeros((1, N),dtype='float32') # sum of the velocity potencial detected by each of the surface elements of the divided sensor
        if angsens: # angle sensitivity
            nS = -1*posSens[:,currentSens] # [3,]
            nS = nS[0:2]
            mnS = np.sqrt(nS[0] ** 2 + nS[1] ** 2) # norm l2
            nS = nS/mnS # detector surface normal
            nS = np.reshape(nS,(2,1)) @ np.ones((1, N),dtype='float32') # [2,N] 
        for kk in range(0, nls):  # For each surface element of the divided sensor
            # Calculate the distance between DVol and posSens
            a1 = posSens[:,currentSens] # center of the sensor [3,]
            a2 = posz[kk] # bias [1,]
            a3 = np.sqrt(a1[0]**2 + a1[1]**2) # center position modulus in x-y plane 
            a4 = np.sqrt(a2**2 + a3**2) # element position modulus in x-y plane
            a5x = (a1[0]/a3*np.cos(a2/a3)-a1[1]/a3*np.sin(a2/a3))*a4 # x coordinate of the detector element 
            a5y = (a1[0]/a3*np.sin(a2/a3)+a1[1]/a3*np.cos(a2/a3))*a4 # y coordinate of the detector element
            posele = np.array([a5x,a5y]) # [2,]
            aux2 = rj - np.reshape(posele,(2,1))@np.ones((1,N)) # [2,N]
            R = np.sqrt(aux2[0, 0:] ** 2 + aux2[1, 0:] ** 2) # [N,]
            # Weight factor: shape factor and angle sensitivity 
            wf = np.ones((1,N),dtype='float32') # [1,N]
            if angsens: # angle sensitivity
                wf = np.abs(np.sum(nS*(-1*aux2),axis=0)/R) # cos(tita) = (nS dot (rd-rj))/(|nS|*|rd-rj|) [N,]
                wf = np.reshape(wf, (1, N)) # [1,N]
            if SF: # shape factor
                # Disc shape:
                dsf = 1/2
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
                delta = np.exp(-1*((2*Ti/tprop)**2))
            else:
                delta = (np.abs(sampleTimes[currentTime] * dt - R / vs)) <= dt / 2 
                delta = delta * 1  # Bool to int
            acum = acum + wf * Ilaser * delta / R  # sum of the velocity potencial detected by each of the surface elements of the divided sensor
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
        Tm2=(np.abs(Tm)<=(dx/(vs*dt)))*1; 
        Tm2=Tm2*Tm
        Tm2=Tm2*(-vs*dt/(2*dx));
        Gpa=convolution_matrix(Tm2,Nt,'same') # GPa is adimensional
        Gpa=sparse.kron(csc_matrix(np.eye(Ns,dtype='float32')),csc_matrix(Gpa,dtype='float32'))
        del Tm, Tm2
        gc.collect()
        A = Gpa@Gs
        A = (Betta/(4*np.pi*vs**2)*DVol/dt**2)*A;  # A is adimensional
        del Gpa, Gs
        gc.collect()
    else:
        A = Gs
        A = (-Betta / (4 * np.pi * rho * Calp) * DVol / dt) * A  # [m^5/(J*s)]    
    
    if MDIR.size != 0: # Detector Impulse Response
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
    
    P0 = P0/np.max(P0)
    P0 = np.reshape(P0,(nx,nx)) 
    
    return P0 

###############################################################################
if __name__ == '__main__':
    
    saveA = False
    
    import time
    start = time.perf_counter()
    
    # Simulation parameters: example for ideal point sensors
    Ns = 15         # number of detectors
    Nt = 1024       # number of time samples
    dx = 115e-6     # pixel size  in the x direction [m]
    nx = 128        # number of pixels in the x direction for a 2-D image region
    Rs = 44e-3      # radius of the circunference where the detectors are placed [m]
    arco = 360      # arc of the circunference where the detectors are placed [°] 
    vs = 1490       # speed of sound [m/s]
    to = 21.5e-6    # initial time [s]
    T = 25e-6       # size of the time windows [s].
    tf = to + T     # final time [s].
    nhie = 0        # Non homogeneous illumnation effect? if nihe == 0 --> is disable

    Ao = createForwMat(Ns,Nt,dx,nx,Rs,arco,vs,to,tf,nhie,1)
   
    end = time.perf_counter()
    print('Time to create the matrix [minutes]',(end - start)/60)
    
    if saveA:
        import scipy as sp
        print('Saving matrix...')
        sp.sparse.save_npz('Ao'+'.npz',Ao)
        print('done!')