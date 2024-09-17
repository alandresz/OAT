from osctck2c import Osctck
from rotmcESP import RotmcESP
from utils import getFilePath
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

TDS2024 = Osctck('USB0::1689::874::C034414::0::INSTR')
ESP300 = RotmcESP('ASRL/dev/ttyUSB0::INSTR')

TDS2024.config(channels = (1,4),
                 triggerSource = 'EXT',
                 triggerLevel = 0.5,
                 triggerSlope = 'RISE',
                 triggerMode = 'NORM',
                 triggerCoup = 'AC',
                 acquisition = 16,
                 vAutoScale = True,
                 chanband = ('ON', 'ON'),
                 chaninv = ('ON', 'OFF' ),
                 chancopu = ('AC', 'AC') )

ESP300.config(axis = 2,
              vel = 2,
              direction = '+',
              setOrigin = True)

# Parámetros de rotación
initAng = 0
endAng = 360
angleStep = 10
currPostion = initAng
arco = endAng-initAng

# Variables de guardado de datos
Rto = 107.13e3 # [ohm]
Rs = (44e-3, 44e-3)#42.625e-3 # [m] # distancia entre el transductor y el centro del eje de rotación
El = 13 # [mJ]
tmed = 34e-6#  # [s]
timebase= 2.5e-6##seconds per division
Navg = 16
sample = 'derenzo'

ltMeas1 = []
ltMeas2 = []
ltAng = []
filePath = getFilePath()

# Comienzo de ciclo de medición
for i in tqdm(range(initAng, endAng, angleStep)):
    # print("\nRealizando medición a {0}°...".format(currPostion))
    meas = TDS2024()
    ltAng.append(currPostion)
    ltMeas1.append(meas[1])
    ltMeas2.append(meas[2])
    currPostion = ESP300(reference = 'REL', rotAngle = angleStep)

# Guardado de datos
sinogram1 = np.array([np.array(i) for i in ltMeas1])
sinogram2 = np.array([np.array(i) for i in ltMeas2])
t = meas[0]
angles = np.array(ltAng)
np.savez(filePath + '.npz', t=t, S1=sinogram1, S2=sinogram2, A=angles, Rto=Rto, Rs=Rs, El=El, Navg=Navg, sample = sample, arco=arco)
print(angles.shape),print(angles)