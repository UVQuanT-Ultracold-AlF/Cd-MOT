from init import *
from MC_base import *
from sys import argv

H = Hamiltonians[int(argv[1])]
det_MOT = float(argv[2])*1e6/hertz_unit + isotope_shifts[112]
det_slower = float(argv[3])*1e6/hertz_unit + isotope_shifts[112]
num_run = int(argv[4])
pol = argv[5]

LASER = {'-' : MOT_and_Slow_Beams , '+'  : MOT_and_Slow_Beams_sig_2, '0' :  MOT_and_Slow_Beams_lin}[pol]

# samp_isotope = fake_samp(112) #rej_samp(func = lambda x : abundance_data[x], rand_x = lambda : rand.randint(106,116), rand_y = lambda : rand.uniform(0,0.2873))
# samp_vel = rej_samp(func = lambda x : capture_pdf(x[0]), rand_x = lambda : [rand.uniform(100/velocity_unit,200/velocity_unit), 0, 0], rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_v0 = rej_samp(func = vel_dist, rand_x = lambda : rand.uniform(np.min(vel_dist_data[:,0]),np.max(vel_dist_data[:,0])), rand_y = lambda : rand.uniform(0,np.max(vel_dist_data[:,1]))) #cdf_samp(capture_cdf, [0,300/velocity_unit]) # rej_samp(func = lambda x : capture_pdf(x), rand_x = lambda : rand.uniform(100/velocity_unit,200/velocity_unit), rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_vt = cdf_samp(transverse_cdf, [-transverse_cutoff, transverse_cutoff]) # rej_samp(func = lambda x : transverse_pdf(abs(x)), rand_x = lambda : rand.uniform(-transverse_cutoff,transverse_cutoff), rand_y = lambda : rand.uniform(0,transverse_pdf(transverse_cutoff)))
samp_angle = rej_samp(rand_x = lambda : rand.uniform(0,2*np.pi))
samp_vel = rej_samp(func = lambda x : x, rand_x = lambda : [next(samp_v0), *((lambda x, a : [x*np.sin(a), x*np.cos(a)])(next(samp_vt), next(samp_angle)))], rand_y = lambda : 0, comp_func = lambda x ,y : abs(np.arctan(np.sqrt(x[1]**2 + x[2]**2)/x[0])) < 75e-3 if abs(x[0]) > 1e-10 else False)
samp_time = cdf_samp(lambda x : norm.cdf(x, 1e-3/time_unit,0.5e-3/time_unit), [0/time_unit, 2.5/time_unit]) # rej_samp(func = lambda x : norm.pdf(x, 1e-3/time_unit,0.5e-3/time_unit), rand_x = lambda : rand.uniform(0,2.5e-3/time_unit), rand_y = lambda : rand.uniform(0,norm.pdf(1e-3/time_unit, 1e-3/time_unit,0.5e-3/time_unit)))
samp_pos = rej_samp(func = lambda r : 1 if sum(map(lambda x : x**2, r)) < (2e-1)**2 else 0, rand_x = lambda : (rand.uniform(-2e-1,2e-1),rand.uniform(-2e-1,2e-1)), rand_y = lambda : 0.5)

MOT_range = np.linspace(-750e6/hertz_unit, 2100e6/hertz_unit,201)

    


dr = 0.11
dv = 2.1/velocity_unit

r = np.arange(-20, 10+dr, 2*dr)
v = np.arange(-250/velocity_unit, 400/velocity_unit, 2*dv)

R, V = np.meshgrid(r, v)


eq = pylcp.rateeq(LASER(det_MOT=det_MOT, det_slower=det_slower),permMagnetsPylcp,H,include_mag_forces=False)
try:
    eq.set_initial_pop(np.array([1,0,0,0]))
except ValueError:
    eq.set_initial_pop(np.array([0.5,0.5,0,0,0,0,0,0]))
eq.generate_force_profile([R, np.zeros(R.shape), np.zeros(R.shape)],
                           [V, np.zeros(V.shape), np.zeros(V.shape)],
                           name='Frad', progress_bar=True)

def single_run(i = None):
    global eq
    if i is not None:
        print(i)
    pos = next(samp_pos)
    vel = next(samp_vel)
    t0 = next(samp_time)
    eq.set_initial_position_and_velocity(np.array([-45.5,pos[0],pos[1]]),vel)
    return eq.evolve_motion([t0, t0 + 25e-3/time_unit], events=[captured_condition,lost_condition,backwards_lost],
                    max_step=m_step)

runs = [single_run(v) for v in range(num_run)]


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

xlim = plt.xlim()
ylim = plt.ylim()

[plt.plot(1e-2*r.r[0],velocity_unit*r.v[0], '-', color = "black" if isCaptured(r) < 1 else "green") for r in runs]

plt.xlim(xlim)
plt.ylim(ylim)

plt.tight_layout()
plt.show()

