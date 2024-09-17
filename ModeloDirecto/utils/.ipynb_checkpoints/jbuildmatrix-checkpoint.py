# General modules
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

# JAX modules
#import jax 
from jax import jit
from jax import numpy as jnp
from jax import grad

# j-wave modules
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.geometry import Sensors, points_on_circle
from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation

###############################################################################
def SensorMaskCartCircleArc(circle_radius, circle_arc, num_sensor_points):
    """
    Matrix with the Ns locations (num_sensor_points) of the sensors arranged 
    on a circunference arc (circle_arc) with a radius circle_radius
    """
    th = np.linspace(0, circle_arc * np.pi / 180, num_sensor_points + 1)
    th = th[0:(len(th) - 1)]  # Angles
    # position of the center of the sensors:
    posSens = np.array([np.cos(th), np.sin(th)])*circle_radius  
    posSens = posSens.astype(np.float32)
    return posSens # (2,Ns)

##############################################################################
# Resampling time
def resamptime(Time,Sensordata,t):
    jS = np.asarray(Sensordata.T)
    tj = np.asarray(Time)
    Ns = jS.shape[0]
    Nt = t.shape[0]
    #pd = jnp.zeros((Ns,Nt)) # [Ns,Nt]
    pd = np.zeros((Ns,Nt)).astype(np.float32) # [Ns,Nt]
    for i1 in range(0,Ns):
        fp=interp1d(tj[:],jS[i1,:])
        aux=fp(t)
        #pd = pd.at[i1,:].set(aux)
        pd[i1,:]=aux
    return pd

##############################################################################
# Resampling time 2j
def resamptime2j(t,S,Time):
    tj = np.asarray(Time)
    jNt = tj.shape[0]
    dt = t[1]-t[0]
    Nto = int(np.abs(tj[0]-t[0])/dt) 
    Ntf = int(np.abs(tj[-1]-t[-1])/dt)
    Dto = np.linspace(tj[0],t[0]-dt,Nto)
    Dtf = np.linspace(t[-1]+dt,tj[-1],Ntf)
    tT = np.concatenate((Dto,t))
    tT = np.concatenate((tT,Dtf))
    Ns = S.shape[0]
    ST = np.concatenate((np.zeros((Ns,len(Dto))),S),axis=1)
    ST = np.concatenate((ST,np.zeros((Ns,len(Dtf)))),axis=1)
    pd = np.zeros((Ns,jNt)).astype(np.float32) # [Ns,jNt]
    for i1 in range(0,Ns):
        fp=interp1d(tT[:],ST[i1,:])
        aux=fp(tj)
        pd[i1,:]=aux
    return jnp.array(pd.T)

##############################################################################
def rotateimage(img,angle):
    
    nx, ny = img.shape
    
    # using cv2.getRotationMatrix2D() to get the rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center=(nx//2,ny//2), angle=angle, scale=1)

    # rotate the image using cv2.warpAffine
    rotated_img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(nx, ny))
    
    return rotated_img

##############################################################################
def cropimage(img,newsize):
    
    nx, ny = img.shape
    mx, my = newsize
    
    i = (nx - mx)//2
    j = (ny - my)//2 
    
    cropped_img = img[i:i+mx,j:j+my]
    
    return cropped_img

##############################################################################
# Resampling time
def resamptime2(x,y,xnew):
    fp=interp1d(x,y)
    ynew = fp(xnew)

    return ynew

##############################################################################
def square_mask(N,nx):
    """
    Generate a 2D binary mask representing a square within a 2D grid.

    """
    x, y = np.mgrid[0:N[0], 0:N[1]]
    maskx = ((x >= N[0]//2 - nx//2) & (x < N[0]//2 + nx//2)).astype(int)
    masky = ((y >= N[1]//2 - nx//2) & (y < N[1]//2 + nx//2)).astype(int)
    return maskx*masky

##############################################################################
def build_matrix(N,dx,vs,Ns,dsa,nx,Nt,to,tf,nhie):
    """
    Model-based Matrix using j-wave -> A: (Ns*Nt,N)
    P = A@P0 # where P: acoustic pressure (Ns*Nt,)

    N: number of pixels in the j-wave 2-D Domain
    dx: pixel size  in the x direction [m]
    vs: speed of sound (homogeneous medium with density = 1000 kg/m3) [m/s]
    Ns: number of detectors
    dsa: radius of the circunference where the detectors are placed [m]
    nx: number of pixels in the x direction for a 2-D image region
    Nt: number of time samples
    to: initial time [s]
    tf: final time [s] 
    nhie: Non homogeneous illumnation effect? if nihe == 0 --> is disable
    
    References:
        [1] A. Hauptmann, et al., "Deep learning in PAT: current approaches 
        and future directions", J. Bio. Opt. 25, p. 112903 (2020).
        [2] A. Stanziola, et al. "j-Wave: An open-source differentiable wave 
        simulator", arXiv (2022).
    """    
    
    # Domain
    jN, jdx = (N, N), (dx, dx)
    domain = Domain(jN, jdx)
    
    # Acoustic medium
    medium = Medium(domain=domain, sound_speed=vs)
    
    # Time
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)
    Time = time_axis.to_array()
    
    # Sensors
    num_sensors = Ns
    x, y = points_on_circle(num_sensors, int(np.round(dsa/dx)), (N//2, N//2))
    sensors_positions = (x, y)
    sensors = Sensors(positions=sensors_positions)
    
    # Illuminaton effect
    if nhie > 0:
        w0 = 0.7e-3; f = -30e-3; l0 = 532e-9; n = 1;
        z0 = np.pi*n*w0**2/l0
        w = np.sqrt(z0*l0/np.pi*((1-nhie/f)**2+(nhie/z0)**2));
    
    # Compile simulation
    @jit
    def compiled_simulator(medium, p0):
        a = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)
        return a
    
    # Initialise system matrix
    #A = jnp.zeros((Ns*Nt,nx**2));
    #t = jnp.linspace(to, tf, Nt)
    A = np.zeros((Ns*Nt,nx**2)).astype(np.float32)
    A = A.T
    t = np.linspace(to, tf, Nt).astype(np.float32)
    val = 1.0
    
    cont = -1; # MM row number
    with tqdm(total=nx**2, leave=True) as pbar:
        for kkx in range(nx):
            for kky in range(nx):
                cont = cont + 1;
            
                # Create initial pressure distribution
                p0 = jnp.zeros((N,N)).astype(jnp.float32);
                corx=N//2-nx//2+kkx;
                cory=N//2-nx//2+kky;
                #p0[corx,cory]=1; # [Pa]
                if nhie > 0:
                    val = np.exp(-((-nx//2*dx+kkx*dx)**2 + (-nx//2*dx+kky*dx)**2)/2/w**2)
                p0 = p0.at[corx,cory].set(val)
        
                # Set the initial condition (PA source term)
                p0 = 1.0 * jnp.expand_dims(p0, -1)
                p0 = FourierSeries(p0, domain)
        
                # Run the jwave simulation
                sensors_data = compiled_simulator(medium, p0)[..., 0] # [Nt,Ns]
        
                # Record row cont
                pd = resamptime(Time,sensors_data,t) # [Ns,Nt]
                #pd = jnp.reshape(pd,(Ns*Nt,1)) # [Ns*Nt,1]
                #pd = np.reshape(pd,(Ns*Nt,1)) # [Ns*Nt,1]
                pd = pd.ravel() # [Ns*Nt,1]
                #A=A.at[cont,:].set(pd)
                A[cont,:] = pd
                
                pbar.update()
                #print('Column: ', cont+1, ' de ', nx**2)
            
    #return np.asarray(A.T)
    return A.T


##############################################################################
def tria(N,dx,vs,dsa,nx,to,tf,pmed):
    """
    The function uses the reciprocity of the wave equation to generate
    a time reversal imaging algorithm.

    N: number of pixels in the j-wave 2-D Domain
    dx: pixel size  in the x direction [m]
    vs: speed of sound (homogeneous medium with density = 1000 kg/m3) [m/s]
    dsa: radius of the circunference where the detectors are placed [m]
    nx: number of pixels in the x direction for a 2-D image region
    to: initial time [s]
    tf: final time [s] 
    pmed: sinogram [Ns x Nt]
    
    References:
        [1] A. Stanziola, et al. "j-Wave: An open-source differentiable wave 
        simulator", arXiv (2022).
    """    
    
    # Measurements parameters
    Ns, Nt = pmed.shape # Ns: number of detectors, Nt: number of time samples
    
    # Domain
    jN, jdx = (N, N), (dx, dx)
    domain = Domain(jN, jdx)
    
    # Acoustic medium
    medium = Medium(domain=domain, sound_speed=vs)
    
    # Time
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)
    Time = time_axis.to_array()
    
    # Sensors
    num_sensors = Ns
    x, y = points_on_circle(num_sensors, int(np.round(dsa/dx)), (N//2, N//2))
    sensors_positions = (x, y)
    sensors = Sensors(positions=sensors_positions)
    
    # Compile simulation
    @jit
    def compiled_simulator(medium, p0):
        simdata = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)
        return simdata
    
    
    def lazy_time_reversal(measurements,smask):
        def mse_loss(p0, measurements):
            p0 = p0.replace_params(p0.params)
            p_pred = compiled_simulator(p0)[..., 0]
            return 0.5 * jnp.sum(jnp.abs((p_pred - measurements)*smask) ** 2)

        # Start from an empty field
        p0 = FourierSeries.empty(domain)

        # Take the gradient of the MSE loss w.r.t. the
        # measured data
        p_grad = grad(mse_loss)(p0, measurements)

        return -p_grad
    
    # Resamplig measurements
    t = np.linspace(to, tf, Nt).astype(np.float32)
    measdata = resamptime2j(t,pmed,Time)
    
    # Square mask
    smask = square_mask(N,nx)

    # Reconstruct initial pressure distribution
    recon_image = lazy_time_reversal(measdata,smask)
    
    return recon_image

##############################################################################
def build_sym_matrix(N,dx,vs,Ns,dsa,nx,Nt,to,tf,thresh,nhie,astr):
    """
    Model-based Matrix using j-wave -> A: (Ns*Nt,N) assuming a OAT system with
    a symmetrical detection subsystem.
    P = A@P0 # where P: acoustic pressure (Ns*Nt,)

    N: number of pixels in the j-wave 2-D Domain
    dx: pixel size  in the x direction [m]
    vs: speed of sound (homogeneous medium with density = 1000 kg/m3) [m/s]
    Ns: number of detectors
    dsa: radius of the circunference where the detectors are placed [m]
    nx: number of pixels in the x direction for a 2-D image region
    Nt: number of time samples
    to: initial time [s]
    tf: final time [s] 
    thresh: threshold the matrix to remove small entries and make it more 
    sparse 10**(-thresh)
    nhie: Non homogeneous illumnation effect? if nihe == 0 --> is disable
    astr: Apply sensor time response? True or False
    
    References:
        [1] A. Hauptmann, et al., "Deep learning in PAT: current approaches 
        and future directions", J. Bio. Opt. 25, p. 112903 (2020).
        [2] A. Stanziola, et al. "j-Wave: An open-source differentiable wave 
        simulator", arXiv (2022).
        [3] N. Awasthi, et al., "Deep Neural Network Based Sinogram 
        Super-resolution and Bandwidth Enhancement for Limited-data PAT",
        IEEE TUFFC. 67, pp. 2660-2673 (2020).
        [4] B. Treeby, et al. "k-Wave: MATLAB toolbox for the simulation and 
        reconstruction of PA  wave-fields", J. Biomed. Opt. 15, 021314 (2010).
    """    
    
    # Domain
    jN, jdx = (N, N), (dx, dx)
    domain = Domain(jN, jdx)
    xg = np.arange(0,N)
    yg = np.arange(0,N)
    Xgrid, Ygrid = np.meshgrid(xg,yg,indexing='ij') # normalized grid position
    
    # Acoustic medium
    medium = Medium(domain=domain, sound_speed=vs)
    
    # Time
    time_axis = TimeAxis.from_medium(medium, cfl=0.3)
    Time = time_axis.to_array()
    jdt = time_axis.dt
    
    # Sensors
    #num_sensors = Ns
    posSens = SensorMaskCartCircleArc(dsa, 360, Ns)
    y = posSens[0,:]/dx + N/2
    x = posSens[1,:]/dx + N/2
    x = x.astype(int)
    y = y.astype(int)
    #x, y = points_on_circle(num_sensors, int(np.round(dsa/dx)), (N//2, N//2))
    sensors_positions = (x, y)
    sensors = Sensors(positions=sensors_positions)
    
    # Illuminaton effect
    if nhie > 0:
        w0 = 0.7e-3; f = -30e-3; l0 = 532e-9; n = 1;
        z0 = np.pi*n*w0**2/l0
        w = np.sqrt(z0*l0/np.pi*((1-nhie/f)**2+(nhie/z0)**2));
    else:
        Ilaser = 1
    
    # Compile simulation
    @jit
    def compiled_simulator(medium, p0):
        a = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)
        return a
    
    # Initialise system matrix
    A = np.zeros((Ns*Nt,nx**2)).astype(np.float32)
    A = A.T
    
    # Ouptut axis time
    t = np.linspace(to, tf, Nt).astype(np.float32)
    val = 1.0
    
    # Create initial pressure distribution
    p0 = jnp.zeros((N,N)).astype(jnp.float32);
    corx=N//2-nx//2;
    cory=N//2-nx//2;
    p0 = p0.at[corx,cory].set(val)
        
    # Set the initial condition (PA source term)
    p0 = 1.0 * jnp.expand_dims(p0, -1)
    p0 = FourierSeries(p0, domain)
        
    # Run the jwave simulation
    print('Running j-wave simulation...')
    sensors_data = compiled_simulator(medium, p0)[..., 0] # [jNt,Ns]
    sd1 = np.asarray(sensors_data)
    #sd1 = sd1.T  # [Ns,jNt] 
    print('done!')
    
    # Apply sensor time response?
    if astr:
        tj = np.asarray(Time)
        jNt = tj.shape[0]
        sd1 = applyDIR2JWS(tj[0],tj[-1],jNt,Ns,sd1)
    
    # Taking advantage of the symmetry of the OAT system
    print('Creating matrix...')
    sd2 = np.zeros(sd1.shape); # [jNt,Ns] 
    sensor_distance0 = np.sqrt((Xgrid[corx,cory]-x)**2 + (Ygrid[corx,cory]-y)**2) # [Ns,] 
    
    cont = -1; # MM row number
    with tqdm(total=nx**2, leave=True) as pbar:
        for kkx in range(nx):
            for kky in range(nx):
                cont = cont + 1;
            
                corx=N//2-nx//2+kkx;
                cory=N//2-nx//2+kky;
                
                r = np.sqrt((Xgrid[corx,cory]-x)**2 + (Ygrid[corx,cory]-y)**2) # [Ns,] 
                r1 = np.abs(r - sensor_distance0) # [Ns,] 
                
                # Threshold the matrix to remove small entries and make it more sparse
                if thresh>0: 
                    r1[abs(r1)<10**(-thresh)] = 0
                
                ind = np.ceil(r1*dx/vs/jdt); # [Ns,] 
                
                for ks in range(0,Ns):
                    sdks = sd1[:,ks]*np.sqrt(sensor_distance0[ks]/r[ks]) # [jNt,Ns]
                    
                    if r[ks] >= sensor_distance0[ks]:
                        sd2[:,ks] = np.roll(sdks,int(ind[ks]),axis=0)
                    else:
                        sd2[:,ks] = np.roll(sdks,int(-ind[ks]),axis=0)
                
                # Illumination effect
                if nhie > 0:
                    Ilaser = np.exp(-((-nx//2*dx+kkx*dx)**2 + (-nx//2*dx+kky*dx)**2)/2/w**2)
                
                # Record row cont
                pd = resamptime(Time,sd2,t) # [Ns,Nt]
                pd = pd.ravel() # [Ns*Nt,1]
                A[cont,:] = pd*Ilaser
                
                pbar.update()
    print('done!')           
    return A.T

##############################################################################
def applyDIR2JWS(to,tf,Nt,Ns,S):
    
    print('Applying detector impulse response to j-wave sinogram...');
    t = np.linspace(to, tf, Nt,dtype=np.float32) # time grid
    ti = t-to
    # Detector impulse response (limited bandwidth)
    #from utils.transducers import V306SUOAmed
    from utils.transducers import getECO_IR
    from scipy.linalg import convolution_matrix
    source_freq = 2.25e6
    mu = ti[Nt//2]#2.04e-6
    sigma = 0.24e-6  
    impP = getECO_IR(ti,source_freq,mu,sigma)
    #impP = V306SUOAmed(tf-to,Nt)
    impP = impP.astype(np.float32)
    impP = impP*-1
    iPM = convolution_matrix(impP,Nt,'same')
    Ss = iPM@S
    print('done!')
    return Ss

##############################################################################
def applyDIR2JWM(to,tf,Nt,Ns,A):
    
    print('Applying detector impulse response to j-wave matrix..');
    t = np.linspace(to, tf, Nt) # time grid
    ti = t-to
    # Detector impulse response (limited bandwidth)
    from utils.transducers import getECO_IR
    #from utils.transducers import V306SUOAmed
    from scipy.linalg import convolution_matrix
    #from scipy import sparse
    #from scipy.sparse import csc_matrix # Column sparse
    source_freq = 2.25e6
    mu = ti[Nt//2]#2.04e-6
    sigma = 0.24e-6  
    impP = getECO_IR(ti,source_freq,mu,sigma)
    #impP = impP*-1
    #impP = V306SUOAmed(tf-to,Nt)
    MDIR = convolution_matrix(impP,Nt,'same')
    MDIR = MDIR.astype(np.float32)
    #An = sparse.kron(np.eye(Ns,dtype='float32'),MDIR)@csc_matrix(A,dtype='float32')
    An = np.kron(np.eye(Ns,dtype='float32'),MDIR)@A
    
    return An

##############################################################################
def createsinogram(Nt,Nti,ang,img,Ng,dg,t_end,Rs,vs,tlen,plot):
    
    """
    Nt:  number of time samples
    Nti: number of time samples for interpolation Nti >> Nt
    ang: angles to be measured (Na,)
    img: image of the sample (400,400)
    Ng: number of pixels of the SPATIAL GRID
    dg: spatial step [m]
    t_end: simulation end time [s]
    Rs: radius of the circunference where the detectors are placed [m]
    vs: speed of sound of the medium surrounding the sample [m/s]
    tlen: transducer length [m]
    plot: plot results? True or False
    """
    
    Na = ang.shape[0]
    nx, ny = img.shape
    newsize = (200,200)
    
    # Sinogram
    Sinogram = np.zeros((Na,Nt))
    
    for i1 in range(0,Na):
    #for i1 in tqdm(range(0,Na)):
        if i1>0:
            imgr = rotateimage(img,np.round(ang[i1],1))
        else:
            imgr = img
            
        imgr = cropimage(imgr,newsize)
            
        t, pr = simulation_one_position(Ng,dg,t_end,Rs,vs,tlen,imgr,plot)
        
        ti = np.linspace(t[0],t[-1],Nti)
        pri = resamptime2(t,pr,ti)
        
        ini = ti.shape[0] - Nt 
        
        Sinogram[i1,:] = pri[ini:]
    
    return Sinogram

##############################################################################
def simulation_one_position(Ng,dg,t_end,Rs,vs,tlen,img,plot):
    
    """
    Ng: number of pixels of the SPATIAL GRID
    dg: spatial step [m]
    t_end: simulation end time [s]
    Rs: distance between sample and detector [m]
    vs: speed of sound of the medium surrounding the sample [m/s]
    tlen: transducer length [m]
    img: image of the sample, max values = 200 x 200
    plot = True or False
    """
      
    # Settings
    #N = (Ng, Ng)
    #N = (int(tlen*2/dg), Ng)
    N = (300, Ng)
    dx = (dg, dg)
    cfl = 0.25

    # Define domain
    domain = Domain(N, dx)

    # Detector parameters
    n = int(tlen/dx[0])
    y = jnp.ones(n, dtype=int) * int(N[1]*0.02)
    x = jnp.arange((N[0] - n)//2, (N[0] + n)//2, 1)
    Transducer_position = (x,y)

    # Sample parameters
    xo = int(Rs/dx[1] + int(N[1]*0.02)) 
    nx, ny = img.shape
    sample_mask = np.zeros(N)
    sample_mask[N[0]//2-ny//2:N[0]//2+ny//2,xo-nx//2:xo+nx//2] = img 
    
    # Set the initial condition (PA source term)
    p0 = 1.0 * jnp.expand_dims(sample_mask, -1)
    p0 = FourierSeries(p0, domain)
         
    # Define medium
    sound_speed = jnp.ones(N)  * vs
    sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain)
    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=20.0)

    # Time axis
    time_axis = TimeAxis.from_medium(medium, cfl=cfl, t_end=t_end)
    t = time_axis.to_array()
   
    if plot:
        # Plots
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        #ticks = [0, 1e3*dx[1]*N[1], -1e3*dx[0]*N[0]/2, 1e3*dx[0]*N[0]/2]
        #speed_graph = ax.imshow(medium.sound_speed.on_grid, extent=ticks, cmap="gray")
        #fig.colorbar(speed_graph, ax=ax, label = "[m/s]")
        ax.scatter(1000*dx[0]*y, 1000*dx[0]*(x - N[0]//2), c="r", marker=".", label="Transducer")
        #ax.set_title("Sound speed")
        ax.legend(loc="lower right")
        ax.set_xlabel("x-position [mm]")
        ax.set_ylabel("y-position [mm]")
        plt.show()
    
    # Simulation
    sensors = Sensors(Transducer_position)
    @jit
    def compiled_simulator():
        return simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors, checkpoint=True)
    
    
    dotdetsignals = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors, checkpoint=True)
    
    # Signal calculation
    det2dsignal = jnp.sum(dotdetsignals[...,0], axis = 1)
    
    return np.asarray(t), np.asarray(det2dsignal)

##############################################################################
def build_matrix2D(N,dx,vs,Ns,Rs,nx,Nt,to,tf,tlen):
    
    # j-Wave simulation parameters
    Ng = 650
    dg = dx
    Nti = 2*Nt
    t_end = tf
    Na = Ns
    arc = 360
    ang = np.linspace(0,arc,Na+1)
    ang = ang[0:-1]  
    plot = False
    
        
    # Initialise system matrix
    A = np.zeros((Ns*Nt,nx**2)).astype(np.float32)
    A = A.T
    val = 1.0
    
    cont = -1; # MM row number
    with tqdm(total=nx**2, leave=True) as pbar:
        for kkx in range(nx):
            for kky in range(nx):
                cont = cont + 1;
            
                # Create initial pressure distribution
                img = np.zeros((N,N)).astype(np.float32);
                corx=N//2-nx//2+kkx;
                cory=N//2-nx//2+kky;
                img[corx,cory]=val; # [Pa]
                
                # Sinogram determination
                S = createsinogram(Nt,Nti,ang,img,Ng,dg,t_end,Rs,vs,tlen,plot)
                S = S.ravel()
        
                # Record row cont
                A[cont,:] = S
                
                pbar.update()

    return A.T

###############################################################################
def createJForwMatdotdet(Ns,Nt,dx,nx,dsa,vs,to,tf,nhie,LBW,saveA): # 
    """Creating Forward Model-based Matrix for point sensors
    """ 
    
    import time
    start = time.perf_counter()
    
    Nd = nx*8
    thresh = 6 
    Aj = build_sym_matrix(Nd,dx,vs,Ns,dsa,nx,Nt,to,tf,thresh,nhie,False)
    if LBW:
        Aj = applyDIR2JWM(to,tf,Nt,Ns,Aj)
    
    end = time.perf_counter()
    print('Time to create the matrix [minutes]',(end - start)/60)
    
    if saveA:
        import scipy as sp
        print('Saving matrix...')
        Aj = sp.sparse.csc_matrix(Aj,dtype='float32')
        sp.sparse.save_npz('Aj'+'.npz',Aj)
        print('done!')
     
    return Aj

###############################################################################
if __name__ == '__main__':
    
    # Set which GPU unit is going to use the program
    #import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    saveA = False
    
    import time
    start = time.perf_counter()
    
    # Simulation parameters
    Rs = 44e-3      # radius of the circunference where the detectors are placed [m]
    Ns = 4#360//10    # number of detectors
    dx = 115e-6     # pixel size  in the x direction [m]
    nx = 128        # number of pixels in the x direction for a 2-D image region
    vs = 1490       # speed of sound [m/s]
    to = 21.5e-6    # initial time [s]
    T = 25e-6       # size of the time windows [s].
    tf = to + T     # final time [s].
    Nt = 1024       # number of time samples
    Nd = nx*8       # number of pixels of the j-wave SPATIAL GRID
    thresh = 0      # threshold the matrix to remove small entries and make it more sparse 10**(-thresh)
    nhie = 0        # Non homogeneous illumnation effect? if nihe == 0 --> is disable
    astr = False    # Apply sensor time response? True or False

    Aj = build_sym_matrix(Nd,dx,vs,Ns,Rs,nx,Nt,to,tf,0,0,False)
    Aj = applyDIR2JWM(to,tf,Nt,Ns,Aj)
    
    end = time.perf_counter()
    print('Time to create the matrix [minutes]',(end - start)/60)
    
    if saveA:
        import scipy as sp
        print('Saving matrix...')
        Aj = sp.sparse.csc_matrix(Aj,dtype='float32')
        sp.sparse.save_npz('Aj'+'.npz',Aj)
        print('done!')