from init import *

from sys import argv

dr = 0.11
dv = 2.1/velocity_unit

r = np.arange(-20, 10+dr, 2*dr)
v = np.arange(-250/velocity_unit, 400/velocity_unit, 2*dv)

R, V = np.meshgrid(r, v)

#upper = 1_309_864.72
upper = 1_309_864.341 # GHz
laser_det = (1_309_863.55 - upper)*1e9/hertz_unit
#Rfull = np.array([sqrt(2)*R, sqrt(2)*R, np.zeros(R.shape)])
#Vfull = np.array([sqrt(2)*V, sqrt(2)*V, np.zeros(V.shape)])

pol = np.array([0,1,1j])

Beams = {'-' : MOT_and_Slow_Beams , '+'  : MOT_and_Slow_Beams_sig_2, '0' :  MOT_and_Slow_Beams_lin}

eq = pylcp.rateeq(Beams[argv[4]](float(argv[1])*1e6/hertz_unit + isotope_shifts[112],float(argv[2])*1e6/hertz_unit + isotope_shifts[112]),permMagnetsPylcp, Hamiltonians[int(argv[3])],include_mag_forces=False)
try:
    eq.set_initial_pop(np.array([1,0,0,0]))
except ValueError:
    eq.set_initial_pop(np.array([0.5,0.5,0,0,0,0,0,0]))
eq.generate_force_profile([R, np.zeros(R.shape), np.zeros(R.shape)],
                           [V, np.zeros(V.shape), np.zeros(V.shape)],
                           name='Frad', progress_bar=True)

fig, ax = plt.subplots()
im = ax.imshow(eq.profile['Frad'].F[0]*(1 + 0*hbar*k*91e6*2*np.pi*2*np.pi/1.89301e-25/1e3), origin='lower',
           extent=(np.amin(r)/100, np.amax(r)/100,
                   np.amin(v*velocity_unit)-dv/2, np.amax(v*velocity_unit)-dv/2),
           aspect='auto', cmap='RdYlBu')
# plt.text(-0.24,390, "$\Delta = -8.7\\;\\Gamma$\n$I = 0.3\\;I_{sat}$", verticalalignment='top', color='w')
ax.set_xlabel("Position [m]")
ax.set_ylabel("Velocity [m/s]")
cb1 = plt.colorbar(im)
cb1.set_label("Force [$\hbar k\Gamma$]")


vel_res = float(argv[5])

vel = np.arange(vel_res/velocity_unit,400/velocity_unit + vel_res/velocity_unit,vel_res/velocity_unit)
def run(v):
    print(f"{v*velocity_unit} m/s",end="\r")
    try:
        eq.set_initial_pop(np.array([1,0,0,0]))
    except ValueError:
        eq.set_initial_pop(np.array([0.5,0.5,0,0,0,0,0,0]))
    eq.set_initial_position_and_velocity(np.array([-45.5,0,0]),np.array([v,0,0]))
    return eq.evolve_motion([0., 25e-3/time_unit], events=[captured_condition,lost_condition,backwards_lost],
                      max_step=m_step)
runs = [run(v) for v in vel]

xlim = plt.xlim()
ylim = plt.ylim()

[plt.plot(1e-2*r.r[0],velocity_unit*r.v[0], '-', color = "black" if isCaptured(r) < 1 else "green") for r in runs]

plt.xlim(xlim)
plt.ylim(ylim)

plt.tight_layout()
plt.show()
