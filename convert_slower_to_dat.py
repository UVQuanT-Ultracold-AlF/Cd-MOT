import numpy as np
from sys import argv
from init import *

data = np.load(f"{argv[1]}.npz")

header = f"Slower Monte-Carlo scan\ns_MOT[xy,z]=[{MOT_s_xy},{MOT_s_z}], s_slower={slower_s}\nslower_angle=0, start_pos=[-10,0,0]\nInitial parameters:\nv_x=ChirpedSlowing_112_initialDistribution.dat, v_t=norm(mu=0,std=34m/s), Off axis cutoff=75 mrad\nPos=uniform(r^2 < (2 mm)^2), time=norm(mu=1ms,std=0.5ms)\nDelta_MOT=-175MHz, Frequencies relative to Cd112\nNumber of runs={data['MC_RUNS']}\nSlower_detuning [MHz],Signal with sigma-, sigma+, lin pol"

np.savetxt(f'{argv[1]}.dat',np.column_stack((data['slower']*hertz_unit/1e6,*data['data'])),fmt = '%.5e',header=header)
