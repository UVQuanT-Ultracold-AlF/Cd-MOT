from init import *
from MC_base import *
import matplotlib.pyplot as plt
from sys import argv
import functools

# samp_isotope = fake_samp(112) #rej_samp(func = lambda x : abundance_data[x], rand_x = lambda : rand.randint(106,116), rand_y = lambda : rand.uniform(0,0.2873))
# samp_vel = rej_samp(func = lambda x : capture_pdf(x[0]), rand_x = lambda : [rand.uniform(100/velocity_unit,200/velocity_unit), 0, 0], rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_v0 = rej_samp(func = vel_dist, rand_x = lambda : rand.uniform(np.min(vel_dist_data[:,0]),np.max(vel_dist_data[:,0])), rand_y = lambda : rand.uniform(0,np.max(vel_dist_data[:,1]))) #cdf_samp(capture_cdf, [0,300/velocity_unit]) # rej_samp(func = lambda x : capture_pdf(x), rand_x = lambda : rand.uniform(100/velocity_unit,200/velocity_unit), rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_vt = cdf_samp(transverse_cdf, [-transverse_cutoff, transverse_cutoff]) # rej_samp(func = lambda x : transverse_pdf(abs(x)), rand_x = lambda : rand.uniform(-transverse_cutoff,transverse_cutoff), rand_y = lambda : rand.uniform(0,transverse_pdf(transverse_cutoff)))
# samp_vt = fake_samp(0)
samp_angle = rej_samp(rand_x = lambda : rand.uniform(0,2*np.pi))
vel_angle_cutoff = 75e-3
vel_angle_cutoff = np.arctan(0.2/45.5)
samp_vel = rej_samp(func = lambda x : x, rand_x = lambda : [next(samp_v0), *((lambda x, a : [x*np.sin(a), x*np.cos(a)])(next(samp_vt), next(samp_angle)))], rand_y = lambda : 0, comp_func = lambda x ,y : abs(np.arctan(np.sqrt(x[1]**2 + x[2]**2)/x[0])) < vel_angle_cutoff if abs(x[0]) > 1e-10 else False)
samp_time = cdf_samp(lambda x : norm.cdf(x, 1e-3/time_unit,0.5e-3/time_unit), [0/time_unit, 2.5/time_unit]) # rej_samp(func = lambda x : norm.pdf(x, 1e-3/time_unit,0.5e-3/time_unit), rand_x = lambda : rand.uniform(0,2.5e-3/time_unit), rand_y = lambda : rand.uniform(0,norm.pdf(1e-3/time_unit, 1e-3/time_unit,0.5e-3/time_unit)))
samp_pos = rej_samp(func = lambda r : 1 if sum(map(lambda x : x**2, r)) < (2e-1)**2 else 0, rand_x = lambda : (rand.uniform(-2e-1,2e-1),rand.uniform(-2e-1,2e-1)), rand_y = lambda : 0.5)
samp_pos = fake_samp([0,0])

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
    try:
        eq.set_initial_pop(np.array([1., 0., 0., 0.]))
    except ValueError: # Quick and dirty solution to detect the two fermionic hamiltonians
        eq.set_initial_pop(np.array([0.5, 0.5, 0., 0., 0., 0., 0., 0.]))   
    eq.set_initial_position_and_velocity(np.array([-10,pos[0],pos[1]]),vel)
    sol = eq.evolve_motion([t0, t0 + 25e-3/time_unit], events=[captured_condition,lost_condition,backwards_lost],
                    max_step=m_step)
    return 0 if isCaptured(sol) < 0 else 1/MC_RUNS

def MC_run(args):
    global progress
    with progress.get_lock():
        progress.value += 1
        if progress.value % 100 == 0:
            print(f"{progress.value}/{total_runtime}: {100*progress.value/total_runtime:.2f}%")
    det = args[1]
    h = args[0]
    LASER = args[2]
    eq = pylcp.rateeq(LASER(det, -700e6/hertz_unit + isotope_shifts[112]),permMagnetsPylcp,Hamiltonians[h], include_mag_forces=False)
    return single_run(eq)
def plot_MOT(raw_data):
    
    d_slower = raw_data["data"][0].T
    d_MOT = raw_data["data"][1].T
    MOT_range = raw_data["MOT_Scan"]-isotope_shifts[112]
    MC_RUNS = raw_data["MC_RUNS"]
    H = Hamiltonians.keys()
    
    fig, ax = plt.subplots()
    tbp = None
    e = None
    for h,d in zip(H,d_slower):
        if tbp is None:
            e = abundance_data[h]**2*d/MC_RUNS
            tbp = abundance_data[h]*d
        else:
            tbp += abundance_data[h]*d
            e += abundance_data[h]**2*d/MC_RUNS
        ax.errorbar(MOT_range*hertz_unit/1e6, abundance_data[h]*d, fmt="C0--", yerr=abundance_data[h]*np.sqrt(d/MC_RUNS), capsize=4)
        ax.text((MOT_range*hertz_unit/1e6)[np.argmax(d)], abundance_data[h]*np.max(d),f"Cd$_{{{h}}}$", color="C0") if np.max(d) > 1e-3 else None
    l0 = ax.errorbar(MOT_range*hertz_unit/1e6,tbp,fmt="C0-",capsize=4,yerr=np.sqrt(e))
    tbp = None
    e = None
    for h,d in zip(H,d_MOT):
        if tbp is None:
            e = abundance_data[h]**2*d/MC_RUNS
            tbp = abundance_data[h]*d
        else:
            tbp += abundance_data[h]*d
            e += abundance_data[h]**2*d/MC_RUNS
        ax.errorbar(MOT_range*hertz_unit/1e6, abundance_data[h]*d, fmt="C1--", yerr=abundance_data[h]*np.sqrt(d/MC_RUNS),capsize=4)
        ax.text((MOT_range*hertz_unit/1e6)[np.argmax(d)], abundance_data[h]*np.max(d),f"Cd$_{{{h}}}$", color="C1") if np.max(d) > 1e-3 else None
    l1 = ax.errorbar(MOT_range*hertz_unit/1e6,tbp,fmt= "C1",capsize=4,yerr=np.sqrt(e))
    ax.grid()
    ax.set_ylabel("MOT signal [a.u.]")
    ax.set_xlabel("Detuning - Cd$_{112}$ [MHz]")
    fig.tight_layout()
    
    names = ["Slowed", "Unslowed"]
    ax.legend([l0,l1], names)
    
    plt.show()

def init_worker(pgr, t_r, MC_Runs):
    global progress, total_runtime, MC_RUNS
    progress = pgr
    total_runtime = t_r
    MC_RUNS = MC_Runs


if __name__ == "__main__":
    __spec__ = None

    # LASER_NOSLOW = MOT_Beams
    # MOT_range = np.linspace(-750e6/hertz_unit, 2100e6/hertz_unit,201)
    # MOT_range = np.linspace(isotope_shifts[112]-1050e6/hertz_unit, isotope_shifts[112]-700e6/hertz_unit,20)
    MOT_range = [isotope_shifts[112]-175e6/hertz_unit]
    # lasers = [MOT_and_Slow_Beams_lin_timed2, MOT_Beams]
    lasers = [MOT_and_Slow_Beams_timed2, MOT_Beams]
    # Hams = Hamiltonians
    Hams = {112 : Hamiltonians[112]}
    import multiprocessing as mp
    progress = mp.Value('i', 0)
    
    params = lambda : np.array(np.meshgrid(list(Hams.keys()), MOT_range, lasers, list(range(MC_RUNS)), indexing='ij')).T.reshape([-1,4])
    
    total_runtime = functools.reduce(lambda x,y : x*y, [MC_RUNS, len(lasers), len(MOT_range), len(list(Hams.keys()))])

    with mp.Pool(processes=MC_CORES, initializer=init_worker, initargs=(progress,total_runtime, MC_RUNS)) as pool:
        data =  np.sum(np.array(pool.map(MC_run, params())).reshape([MC_RUNS, len(lasers), len(MOT_range), len(list(Hams.keys()))]), axis = 0)
    
    
    np.savez("out_MOT.npz", data = data, MOT_Scan = MOT_range, MC_RUNS = MC_RUNS)
