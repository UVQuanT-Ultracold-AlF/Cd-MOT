from init import *

from sys import argv

eq = pylcp.rateeq(MOT_Beams_push_beam(float(argv[1])*1e6/hertz_unit + isotope_shifts[112],float(argv[2])*1e6/hertz_unit + isotope_shifts[112],0),permMagnetsPylcp, Hamiltonians[int(argv[3])],include_mag_forces=False)

i = 0
t_eval = np.arange(0e-6/time_unit,3e-3/time_unit, 0.01e-3/time_unit)
def run():
    global i
    i += 1
    print(i,end='\r')
    try:
        eq.set_initial_pop(np.array([1,0,0,0]))
    except ValueError:
        eq.set_initial_pop(np.array([0.5,0.5,0,0,0,0,0,0]))
    eq.set_initial_position_and_velocity(np.array([-0.04,0,0]),np.array([-4/velocity_unit,0/velocity_unit,-0/velocity_unit]))
    return eq.evolve_motion([0., 3e-3/time_unit], t_eval = t_eval,
                      max_step=1e-6/time_unit,progress_bar=True,random_recoil=True)

r = run()

r_avg = np.mean(np.array([r.r]), axis= 0)

plt.ion()
plt.figure()
plt.plot(r.t*time_unit*1e3, r_avg[0]*10,'rx-')
plt.grid()
plt.show()
plt.figure()
plt.plot(t_eval*time_unit*1e3, r_avg[1]*10,'gx-')
plt.plot(t_eval*time_unit*1e3, r_avg[2]*10,'bx-')
plt.grid()
plt.show()