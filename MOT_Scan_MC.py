from init import *
from MC_base import *
import matplotlib.pyplot as plt
from sys import argv

# samp_isotope = fake_samp(112) #rej_samp(func = lambda x : abundance_data[x], rand_x = lambda : rand.randint(106,116), rand_y = lambda : rand.uniform(0,0.2873))
# samp_vel = rej_samp(func = lambda x : capture_pdf(x[0]), rand_x = lambda : [rand.uniform(100/velocity_unit,200/velocity_unit), 0, 0], rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_v0 = rej_samp(func = vel_dist, rand_x = lambda : rand.uniform(np.min(vel_dist_data[:,0]),np.max(vel_dist_data[:,0])), rand_y = lambda : rand.uniform(0,np.max(vel_dist_data[:,1]))) #cdf_samp(capture_cdf, [0,300/velocity_unit]) # rej_samp(func = lambda x : capture_pdf(x), rand_x = lambda : rand.uniform(100/velocity_unit,200/velocity_unit), rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_vt = cdf_samp(transverse_cdf, [-transverse_cutoff, transverse_cutoff]) # rej_samp(func = lambda x : transverse_pdf(abs(x)), rand_x = lambda : rand.uniform(-transverse_cutoff,transverse_cutoff), rand_y = lambda : rand.uniform(0,transverse_pdf(transverse_cutoff)))
samp_angle = rej_samp(rand_x = lambda : rand.uniform(0,2*np.pi))
samp_vel = rej_samp(func = lambda x : x, rand_x = lambda : [next(samp_v0), *((lambda x, a : [x*np.sin(a), x*np.cos(a)])(next(samp_vt), next(samp_angle)))], rand_y = lambda : 0, comp_func = lambda x ,y : abs(np.arctan(np.sqrt(x[1]**2 + x[2]**2)/x[0])) < 75e-3 if abs(x[0]) > 1e-10 else False)
samp_time = cdf_samp(lambda x : norm.cdf(x, 1e-3/time_unit,0.5e-3/time_unit), [0/time_unit, 2.5/time_unit]) # rej_samp(func = lambda x : norm.pdf(x, 1e-3/time_unit,0.5e-3/time_unit), rand_x = lambda : rand.uniform(0,2.5e-3/time_unit), rand_y = lambda : rand.uniform(0,norm.pdf(1e-3/time_unit, 1e-3/time_unit,0.5e-3/time_unit)))
samp_pos = rej_samp(func = lambda r : 1 if sum(map(lambda x : x**2, r)) < (2e-1)**2 else 0, rand_x = lambda : (rand.uniform(-2e-1,2e-1),rand.uniform(-2e-1,2e-1)), rand_y = lambda : 0.5)

MOT_range = np.linspace(-750e6/hertz_unit, 2100e6/hertz_unit,201)
LASER = MOT_and_Slow_Beams
LASER_NOSLOW = MOT_Beams

if __name__ == "__main__":
    MC_RUNS = int(argv[1])
    MC_CORES = int(argv[2])
else:
    MC_RUNS = 1
    MC_CORES = 1

def single_run(eq):
    pos = next(samp_pos)
    vel = next(samp_vel)
    t0 = next(samp_time)
    eq.set_initial_pop(np.array([1,0,0,0]))
    eq.set_initial_position_and_velocity(np.array([-45.5,pos[0],pos[1]]),vel)
    sol = eq.evolve_motion([0., 25e-3/time_unit], events=[captured_condition,lost_condition,backwards_lost],
                    max_step=m_step)
    return 0 if isCaptured(sol) < 0 else 1/MC_RUNS

def MC_run(args):
    det = args[1]
    h = args[0]
    print(f"{det*hertz_unit/1e6:.2f} MHz",end='\r')
    eq = pylcp.rateeq(LASER(det, -700e6/hertz_unit + isotope_shifts[112]),permMagnetsPylcp,Hamiltonians[h], include_mag_forces=False)
    return single_run(eq)
def plot_MOT(raw_data):
    
    fig, ax = plt.subplots()
    ax.plot()
    pass

params = np.array(np.meshgrid(list(Hamiltonians.keys()), MOT_range, indexing='ij')).T.reshape([-1,2])

if __name__ == "__main__":
    __spec__ = None
    import multiprocessing as mp
    progress = mp.Value('i', 0)
    with mp.Pool(processes=16) as pool:
        data = pool.map_async(MC_run, params)    
        noslow_data = pool.map(MC_run_noslow, params)
        data = data.get()
    
    
    np.savez("out_MOT.npz", data = data, noslow_data = noslow_data, MOT_Scan = MOT_range, MC_RUNS = MC_RUNS)