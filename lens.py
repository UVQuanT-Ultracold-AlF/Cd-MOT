from init import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as clr
import matplotlib.cm as cm

bw = 1
laserbeam = pylcp.laserBeams(
    [
    {'kvec':np.array([ 0.115, 0., 1.]), 'pol': np.array([0., 1., 0.]), 'delta':-125e6/hertz_unit, 's':1.5*slower_s,'wb':slower_beam_width, 'pol_coord' : 'cartesian'},
    {'kvec':np.array([ 0.115, 0.,-1.]), 'pol': np.array([0., 1., 0.]), 'delta':-125e6/hertz_unit, 's':1.5*slower_s,'wb':slower_beam_width, 'pol_coord' : 'cartesian'},
    {'kvec':np.array([-0.115, 0.,-1.]), 'pol': np.array([0., 1., 0.]), 'delta':-125e6/hertz_unit, 's':1.5*slower_s,'wb':slower_beam_width, 'pol_coord' : 'cartesian'},
    {'kvec':np.array([-0.115, 0., 1.]), 'pol': np.array([0., 1., 0.]), 'delta':-125e6/hertz_unit, 's':1.5*slower_s,'wb':slower_beam_width, 'pol_coord' : 'cartesian'}
    ]
    , pylcp.gaussianBeam
)

def vneg(t,y):
    return y[-6] + 1e-2/velocity_unit

def lost_forwards(t, y):
    return y[-3] - 50 - 2

vneg.terminal = True
lost_forwards.terminal = True;

def getMotion(v0):
    eq = pylcp.rateeq(laserbeam ,pylcp.constantMagneticField([0,0,0]),Hamiltonians[114],include_mag_forces=False,)
    try:
        eq.set_initial_pop(np.array([1., 0., 0., 0.]))
    except ValueError: # Quick and dirty solution to detect the two fermionic hamiltonians
        eq.set_initial_pop(np.array([0.5, 0.5, 0., 0., 0., 0., 0., 0.]))
    eq.set_initial_position_and_velocity([-10.5/cm_unit, 0, 0], v0)
    eq.evolve_motion([0, 50e-3/time_unit], progress_bar=False, events = [vneg, lost_forwards], max_step=1e-4/time_unit, random_recoil = False)
    return [eq.sol.r[0], eq.sol.r[2]]

norm = clr.Normalize(100,200)
cmap = cm.get_cmap('gnuplot')

vs = np.array([[v/velocity_unit, 0, v2/velocity_unit] for v in np.linspace(100,200,10) for v2 in np.linspace(1,8,8)])
i = 0
def run(v):
    global i
    print(i,end='\r')
    i+=1
    return getMotion(v)
res = [run(v) for v in vs]

plt.figure()
[plt.plot(*r, 'x-', c=cmap(norm(np.linspace(100,200,10)[i%10]))) for i,r in enumerate(res)]
plt.colorbar(cm.ScalarMappable(norm = norm, cmap = cmap))
plt.grid()
plt.show()