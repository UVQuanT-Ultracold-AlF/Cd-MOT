from init import *

from sys import argv

dr = 0.05
dv = 3/velocity_unit

r = np.arange(-10, 10+dr, dr)
v = np.arange(-250/velocity_unit, 400/velocity_unit, dv)

R, V = np.meshgrid(r, v)

#upper = 1_309_864.72
upper = 1_309_864.341 # GHz
laser_det = (1_309_863.55 - upper)*1e9/hertz_unit
#Rfull = np.array([sqrt(2)*R, sqrt(2)*R, np.zeros(R.shape)])
#Vfull = np.array([sqrt(2)*V, sqrt(2)*V, np.zeros(V.shape)])

pol = np.array([0,1,1j])

Beams = {'-' : MOT_and_Slow_Beams_timed2 , '+'  : MOT_and_Slow_Beams_sig_2_timed, '0' :  MOT_and_Slow_Beams_lin_timed, 'N' : MOT_Beams}
Magnet = {'weak' : permMagnetsPylcp, 'strong' : permMagnetsPylcpStrong}[argv[6]]

startpos = np.array([-10,0,0])
eq = pylcp.rateeq(Beams[argv[4]](float(argv[1])*1e6/hertz_unit + isotope_shifts[112],float(argv[2])*1e6/hertz_unit + isotope_shifts[112]), Magnet, Hamiltonians[int(argv[3])],include_mag_forces=False)
try:
    eq.set_initial_pop(np.array([1,0,0,0]))
except ValueError:
    eq.set_initial_pop(np.array([0.5,0.5,0,0,0,0,0,0]))
eq.generate_force_profile([R, np.zeros(R.shape), np.zeros(R.shape)],
                           [V, np.zeros(V.shape), np.zeros(V.shape)],
                           name='Frad', progress_bar=True)

fig, (ax2,ax) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 9]}, sharey=True)
im = ax.imshow(eq.profile['Frad'].F[0]*(1 + 0*hbar*k*91e6*2*np.pi*2*np.pi/1.89301e-25/1e3), origin='lower',
           extent=(np.amin(r)/100, np.amax(r)/100,
                   np.amin(v*velocity_unit)-dv/2, np.amax(v*velocity_unit)-dv/2),
           aspect='auto', cmap='RdYlBu')
# plt.text(-0.24,390, "$\Delta = -8.7\\;\\Gamma$\n$I = 0.3\\;I_{sat}$", verticalalignment='top', color='w')
ax.set_xlabel("Position [m]")
ax2.set_ylabel("Velocity [m/s]")
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
    eq.set_initial_position_and_velocity(startpos,np.array([v,0,0]))
    return eq.evolve_motion([0., 25e-3/time_unit], events=[captured_condition,lost_condition,backwards_lost],
                      max_step=m_step)
runs = [run(v) for v in vel]

xlim = plt.xlim()
ylim = plt.ylim()

cap, s = findCaptureVelocityRange_fast(startpos,eq,0)
cap = np.array(cap)*velocity_unit

[plt.plot(1e-2*r.r[0],velocity_unit*r.v[0], '-', color = "black" if isCaptured(r) < 1 else "green") for r in runs]

plt.xlim(xlim)
plt.ylim(ylim)
vs = np.linspace(*ylim,200)
ax2.plot(capture_pdf(vs/velocity_unit),vs)
ax2.invert_xaxis()
if len(cap) > 1:
    ax.hlines([cap[0],cap[1]],[-1,-1],[1,1],linestyles='dashed')
    print(cap)
plt.tight_layout()
plt.show()

if len(cap) <= 1:
    cap = ['N/A','N/A']

header = f"""Trajectory simulations
Polarisation:{argv[4]}, Magnet:{argv[6]}, Velocity resolution:{argv[5]} m/s
MOT detuning:{argv[1]} MHz, Slower detuning:{argv[2]} MHz, Isotope:{argv[3]}
Capture range: [{cap[0]},{cap[1]}] m/s
Traj_id, Captured [1/0], rx[m], ry[m], rz[m], vx[m/s], vy[m/s], vz[m/s], t[s]"""

conv_to_data = lambda i, run: np.column_stack(([i]*len(run.r[0]), [1 if isCaptured(run) > 0 else 0]*len(run.r[0]),*(run.r/100),*(run.v*velocity_unit),(run.t*time_unit)))

np.savetxt(f"Traj_MOT_{argv[1]}_{argv[2]}_{argv[3]}_{argv[4]}_{argv[5]}_{argv[6]}.dat",np.vstack([conv_to_data(i,r) for i,r in enumerate(runs)]),header=header, fmt='%.4e')

header = f"""Force Profile
Polarisation:{argv[4]}, Magnet:{argv[6]}, Velocity resolution:{argv[5]} m/s
MOT detuning:{argv[1]} MHz, Slower detuning:{argv[2]} MHz, Isotope:{argv[3]}
Capture range: [{cap[0]},{cap[1]}] m/s
rx [m], vx [m/s], Fx [hbar*Gamma*k]
"""

np.savetxt(f"Force_profile_{argv[1]}_{argv[2]}_{argv[3]}_{argv[4]}_{argv[5]}_{argv[6]}.dat",np.column_stack((R.flatten()*1e-2, V.flatten()*velocity_unit, eq.profile['Frad'].F[0].flatten())), header = header, fmt='%.4e')