import numpy as np
from sys import argv
from init import *

data = np.load(f"{argv[1]}.npz")

header = f"MOT Monte-Carlo scan\ns_MOT[xy,z]=[{MOT_s_xy},{MOT_s_z}], s_slower={slower_s}\nslower_angle=0, start_pos=[-10,0,0]\nInitial parameters:\nv_x=ChirpedSlowing_112_initialDistribution.dat, v_t=norm(mu=0,std=34m/s), Off axis cutoff=75 mrad\nPos=uniform(r^2 < (2 mm)^2), time=norm(mu=1ms,std=0.5ms)\nDelta_MOT=-175MHz, Frequencies relative to Cd112\nNumber of runs={data['MC_RUNS']}\nMOT detuning [MHz], Unslowed {','.join(map(str,list(Hamiltonians.keys())))}, Slowed {','.join(map(str,list(Hamiltonians.keys())))}, Errors: Unslowed {','.join(map(str,list(Hamiltonians.keys())))}, Slowed {','.join(map(str,list(Hamiltonians.keys())))}"

MOT_Scan = data['MOT_Scan']-isotope_shifts[112]
d = data['data']
e = data['data']
for i,h in enumerate(Hamiltonians.keys()):
    e = np.sqrt(e/data['MC_RUNS'])
    d[0][:,i] *= abundance_data[h]
    e[0][:,i] *= abundance_data[h]
    d[1][:,i] *= abundance_data[h]
    e[1][:,i] *= abundance_data[h]

np.savetxt(f'{argv[1]}.dat',np.column_stack((MOT_Scan*hertz_unit/1e6,*(d[1].T),*(d[0].T),*(e[1].T),*(e[0].T))),fmt = '%.5e',header=header)
