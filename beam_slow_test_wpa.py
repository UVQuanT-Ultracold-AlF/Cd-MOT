from init import *
import random as rand
from multiprocessing import Pool, Value
import cProfile
import functools as ft
import matplotlib.gridspec as gridspec

gt = lambda x, y : ft.reduce(lambda a, b : a and b, [i >= j for i,j in zip(x,y)])

def rej_samp(func = lambda _ : 1, rand_x = lambda : rand.uniform(0,1), rand_y = lambda : 0, comp_func = lambda x, y : x >= y):
    while True:
        x, y = rand_x(), rand_y()
        # print (x, y, func(x))
        if (comp_func(func(x), y)):
            yield x

def fake_samp(val):
    while True:
        yield val
        
def cdf_samp(cdf, valrange, randrange = None):
    if randrange is None:
        randrange = [cdf(valrange[0]), cdf(valrange[1])]
    while True:
        rand_prop = rand.uniform(*randrange)
        yield bisect(lambda x : cdf(x) - rand_prop,*valrange, xtol = 1e-2/velocity_unit)

abundance_data[107] = 0
abundance_data[109] = 0
abundance_data[115] = 0

# std of 34 corresponds to FWHM of ~80
transverse_pdf = lambda x : norm.pdf(x, 0/velocity_unit, 34/velocity_unit)
transverse_cdf = lambda x : norm.cdf(x, 0/velocity_unit, 34/velocity_unit)
transverse_cutoff = np.tan(35e-3)*200/velocity_unit

samp_isotope = fake_samp(114) # rej_samp(func = lambda x : abundance_data[x], rand_x = lambda : rand.randint(106,116), rand_y = lambda : rand.uniform(0,0.2873))
# samp_vel = rej_samp(func = lambda x : capture_pdf(x[0]), rand_x = lambda : [rand.uniform(100/velocity_unit,200/velocity_unit), 0, 0], rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_v0 = cdf_samp(capture_cdf, [0,300/velocity_unit]) # rej_samp(func = lambda x : capture_pdf(x), rand_x = lambda : rand.uniform(100/velocity_unit,200/velocity_unit), rand_y = lambda : rand.uniform(0,capture_pdf(mean)))
samp_vt = cdf_samp(transverse_cdf, [-transverse_cutoff, transverse_cutoff]) # rej_samp(func = lambda x : transverse_pdf(abs(x)), rand_x = lambda : rand.uniform(-transverse_cutoff,transverse_cutoff), rand_y = lambda : rand.uniform(0,transverse_pdf(transverse_cutoff)))
samp_angle = rej_samp(rand_x = lambda : rand.uniform(0,2*np.pi))
samp_vel = rej_samp(func = lambda x : x, rand_x = lambda : [next(samp_v0), *((lambda x, a : [x*np.sin(a), x*np.cos(a)])(next(samp_vt), next(samp_angle)))], rand_y = lambda : 0, comp_func = lambda x ,y : abs(np.arctan(np.sqrt(x[1]**2 + x[2]**2)/x[0])) < 75e-3 if abs(x[0]) > 1e-10 else False)
samp_time = cdf_samp(lambda x : norm.cdf(x, 1e-3/time_unit,0.5e-3/time_unit), [0/time_unit, 2.5/time_unit]) # rej_samp(func = lambda x : norm.pdf(x, 1e-3/time_unit,0.5e-3/time_unit), rand_x = lambda : rand.uniform(0,2.5e-3/time_unit), rand_y = lambda : rand.uniform(0,norm.pdf(1e-3/time_unit, 1e-3/time_unit,0.5e-3/time_unit)))
samp_pos = rej_samp(func = lambda r : 1 if sum(map(lambda x : x**2, r)) < (2e-1)**2 else 0, rand_x = lambda : (rand.uniform(-2e-1,2e-1),rand.uniform(-2e-1,2e-1)), rand_y = lambda : 0.5)

magnet_data = np.loadtxt("./csv/RingMagnet_BzProfile.csv",delimiter="\t")
def get_interpolator(pos):
    def interpolate_magnet(R):
        nonlocal pos
        R = R - pos
        x = abs(R[0])*10
        prevpos = None
        prevB = None
        
        for p, Bz in magnet_data:
            if(x < p):
                break
            prevpos = p
            prevB = Bz
        else:
            return [0,0,0]

        if prevpos is None:
            return [0,0,0]

        B = prevB + (x - prevpos)*(Bz - prevB)/(p - prevpos)
        
        B = B*10**(-4)*consts.value('Bohr magneton')/(10**6*91*consts.h)

        return [B,0,0]
    return interpolate_magnet

def No_Beams(*args, **kwargs):
    return pylcp.laserBeams([], beam_type=pylcp.gaussianBeam)

def Slow_Beam(det_slower, *args, pol, **kwargs):
    # pol /= sum(map(lambda x : x*np.conj(x), pol))
    return pylcp.laserBeams([
        {'kvec':np.array([-1, 0., 0.]), 'pol': pol, 'delta':det_slower, 's':slower_s,'wb':slower_beam_width, 'pol_coord':'cartesian'}
    ], beam_type=pylcp.gaussianBeam)

def zero_condition(t,y):
    d = ((y[-2] + y[-3])**2 + (y[-2] - y[-3])**2)/2
    return d - 0.5**2 # sum(map(lambda x : x**2, y[-3:])) - 1

def lost_forwards(t, y):
    return y[-3] - 2

def sideways_lost(t, y):
    return sum(map(lambda x : x**2, y[-2:])) - 2

def losing_backwards(t, y):
    return y[-6] + 1e-2/velocity_unit

lost_forwards.terminal = True
zero_condition.terminal = True
sideways_lost.terminal = True
losing_backwards.terminal = True


PROCNUM = 16
BEAM = Slow_Beam
RECOIL = False

slower_magnet = pylcp.magField(get_interpolator([-10.5/cm_unit,0,0]))

def evolve_beam_vel(v0, ham,magnets = slower_magnet,lasers = BEAM, laserargs = {'det_slower' : -175e6/hertz_unit, 'pol' : -1}, time = 0, r = [0,0]):
    eq = pylcp.rateeq(lasers(**laserargs),magnets, ham,include_mag_forces=False,)
    try:
        eq.set_initial_pop(np.array([1., 0., 0., 0.]))
    except ValueError: # Quick and dirty solution to detect the two fermionic hamiltonians
        eq.set_initial_pop(np.array([0.5, 0.5, 0., 0., 0., 0., 0., 0.]))
    eq.set_initial_position_and_velocity([-45.5/cm_unit,r[0],r[1]], v0)
    try:
        eq.evolve_motion([time, time + 50e-3/time_unit], events=[zero_condition, lost_forwards, sideways_lost, losing_backwards], progress_bar=False, max_step = 1e-3/time_unit, random_recoil = RECOIL)
    except ValueError:
        return [None]*5
    # Rejection conditions
    # if(eq.sol.v[0][-1] < 0):
    #     return -1,-1,-1, eq.sol.v[:,-1], eq.sol.r[:,-1]
    
    # if eq.sol.r[0][-1] > 2:
    #     return -2,-2,-2, eq.sol.v[:,-1], eq.sol.r[:,-1]

    if zero_condition(0, eq.sol.r[:,-1]) > 1e-3:
        return -3,-3,-3, eq.sol.v[:,-1], [eq.sol.r[:,0], eq.sol.r[:,-1]]

    return eq.sol.v[0][-1], eq.sol.t, np.sqrt(eq.sol.v[1][-1]**2 + eq.sol.v[2][-1]**2), eq.sol.v[:,-1], [eq.sol.r[:,0] ,eq.sol.r[:,-1]]

def init_worker(pgr):
    global progress
    progress = pgr

MC_runtime = 1_000_000

def MC_run(wpa_angle):
    global progress
    pol = wpa_angle*180/np.pi
    #p = np.array([0, np.cos(wpa_angle), 1j*np.cos(wpa_angle)])
    p = np.array([0,np.exp(1j*wpa_angle),np.exp(-1j*wpa_angle)])
    
    with progress.get_lock():
        progress.value += 1
        cached_progress = progress.value
    if cached_progress % 100 == 0:
        print(f"{cached_progress}/{MC_runtime}: {100*cached_progress/MC_runtime:.2f}%", end = '\r')
    v0 = next(samp_vel)
    # init_speeds.append(v0)
    speed, time, trans_speed, final_speed, poss = evolve_beam_vel(v0, Hamiltonians[next(samp_isotope)], laserargs={'det_slower' : -700e6/hertz_unit, 'pol' : p}, time = next(samp_time), r = next(samp_pos))
    if speed is None:
        return [None]*7
    if (speed < 0):
        return (v0, None, None, None, final_speed, poss, pol)
    # speeds.append(speed)
    # times.append(time)
    return (v0, speed, [time[0], time[-1]], trans_speed, final_speed, poss, pol)

is_not_none = lambda x : not x is None
pre_none_check = lambda x : not x[1] is None

if __name__ == "__main__":
    __spec__ = None
    
    progress = Value('i', 0)

    # speeds = []
    # times = []
    # init_speeds = []
    # for i in range(MC_runtime):
    #     print(f"{i+1}/{MC_runtime}: {100*(i+1)/MC_runtime:.2f}%", end = '\r')
    #     v0 = next(samp_vel)
    #     init_speeds.append(v0)
    #     speed, time = evolve_beam_vel(v0, Hamiltonians[next(samp_isotope)], laserargs={'det_slower' : -607.5e6/hertz_unit}, angle = next(samp_angle), time = next(samp_time), r = next(samp_pos))
    #     if (speed < 0):
    #         continue
    #     speeds.append(speed)
    #     times.append(time)
    WPA_step = 16
    WPAs = list(np.linspace(0, np.pi, WPA_step))*(MC_runtime//WPA_step);
    with Pool(processes=PROCNUM, initializer=init_worker, initargs=(progress,)) as pool:
        res = pool.map(MC_run, WPAs)
    
    # res = map(MC_run, range(MC_runtime))
    # res = []
    # for i in range(MC_runtime):
    #     cProfile.run('res.append(MC_run(i))')
    # res = list(filter(None, res))

    res = list(filter(lambda x : not x[0] is None, res))

    init_speeds, speeds, times, trans_speeds, final_vels, poss, pols = zip(*res)
    
    no_none_res = list(filter(pre_none_check, res))
    
    init_speeds_no_none, speeds, times, trans_speeds, final_vels, cap_poss, no_none_pols = zip(*no_none_res)
    
    start_times, times = zip(*times)
    start_poss, final_poss = zip(*poss)
    capstart_poss, _ = zip(*cap_poss)
    
    speeds = np.asarray(speeds)
    init_speeds = np.asarray(init_speeds)
    times = np.asarray(times)
    trans_speeds = np.asarray(trans_speeds)
    start_times = np.asarray(start_times)
    start_poss = np.asarray(start_poss)
    final_poss = np.asarray(final_poss)
    init_speeds_no_none = np.asarray(init_speeds_no_none)

    transformed_speeds = np.einsum('i,ij->j',np.array([1,1])/np.sqrt(2), np.array([speeds, trans_speeds]))

    # t_samp, iso_samp, pos_samp = list(map(lambda x : list(map(lambda _ : next(x), range(int(1e4)))), [samp_time, samp_isotope, samp_pos]))
    vel_samp = init_speeds
    t_samp = start_times
    pos_samp = start_poss
    
    plt.figure()
    plt.hist2d(*(np.array(list(map(lambda x : [x[0]*time_unit*1e3 ,x[1]], zip(times, no_none_pols)))).T),bins=[25,WPA_step])
    plt.show()
    # fig = plt.figure(figsize= [12, 8])
    
    # gs = gridspec.GridSpec(2, 1, figure=fig)
    
    # gs0 = gs[0].subgridspec(1,3)
    # # ax1 = fig.add_subplot(gs0[:,0])
    # ax2 = fig.add_subplot(gs0[:,0])
    # ax3 = fig.add_subplot(gs0[:,1])
    # axcappos = fig.add_subplot(gs0[:,2])
    
    # gs1 = gs[1].subgridspec(1,3)

    # axtime = fig.add_subplot(gs1[:,0])
    # # axisotope = fig.add_subplot(gs1[:,1])
    # axpos = fig.add_subplot(gs1[:,1])
    # # axvel= fig.add_subplot(gs1[:,2])
    
    # gsvel = gs1[:,2].subgridspec(6,6)
    # axvel = fig.add_subplot(gsvel[1:,:-1])
    # axvx = fig.add_subplot(gsvel[0,:-1], sharex = axvel)
    # axvt = fig.add_subplot(gsvel[1:,-1], sharey = axvel)
        
    # # fig, ax = plt.subplots(figsize = [12, 8])
    # # plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
    # # ax1.hist(init_speeds[:,0]*velocity_unit, bins = np.linspace(-4,300,305))

    # # # ax2.hist(transformed_speeds*velocity_unit, bins = np.linspace(-4,300,305), label = "Transformed final velocities")
    # # ax1.set_ylabel("Initial Counts [1]")
    # # ax1.set_xlabel("$v_x$ [m/s]")
    # # ax.legend()
    # textstr = '\n'.join([f"Runs: {MC_runtime}", f"Success: {init_speeds[:,0].size}", f"Measured: {times.size}"])
    # props = dict(boxstyle='round')
    # props2 = dict(boxstyle='round', fc = "white", alpha = 0.75)
    # props = props2
    # ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    
    # # plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
    # ax2.hist(init_speeds_no_none[:,0]*velocity_unit, bins = np.linspace(-4,300,305), label = "Initial velocities")
    # ax2.hist(speeds*velocity_unit, bins = np.linspace(-4,300,305), label = "Final velocities")
    # # ax2.hist(transformed_speeds*velocity_unit, bins = np.linspace(-4,300,305), label = "Transformed final velocities")
    # ax2.set_ylabel("Counts [1]")
    # ax2.set_xlabel("$v_x$ [m/s]")
    # ax2.legend()
    # # textstr = '\n'.join([f"Runs: {MC_runtime}", f"Measured: {times.size}"])
    # # props = dict(boxstyle='round')
    # # ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    
    # # plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
    # ax3.hist(times*time_unit*1e3, bins = 100)
    # ax3.set_ylabel("Counts [1]")
    # ax3.set_xlabel("Time of flight [ms]")


    # # axisotope.hist(iso_samp, bins = [106,107,108,109,110,111,112,113,114,115,116,117])
    # # axisotope.set_xlabel("Isotope")
    # # axisotope.set_ylabel("Count [1]")
    
    # axtime.hist(np.array(t_samp)*time_unit*1e3, bins = 100)
    # axtime.set_xlabel("Time [ms]")
    # axtime.set_ylabel("Count [1]")
    
    
    # axpos.hist2d(*np.array(pos_samp)[:,1:].T, bins = [51, 51])
    # axpos.set_xlabel("y [cm]")
    # axpos.set_ylabel("z [cm]")
    # axpos.text(0.05, 0.95, "Start pos for sampled atoms", transform=axpos.transAxes, fontsize=14, verticalalignment='top', bbox = props2)
    
    
    # axcappos.hist2d(*np.array(capstart_poss)[:,1:].T, bins = [51, 51])
    # axcappos.set_xlabel("y [cm]")
    # axcappos.set_ylabel("z [cm]")
    # axcappos.text(0.05, 0.95, "Start pos for measured atoms", transform=axcappos.transAxes, fontsize=14, verticalalignment='top', bbox = props2)
    
    # init_speeds_transf = np.array(list(map(lambda x: [x[0], np.sqrt(x[1]**2 + x[2]**2)], init_speeds)))
    # _, xbins, ybins, _ = axvel.hist2d(*(init_speeds_transf.T*velocity_unit), bins = [51, 51])
    # axvel.set_xlabel("$v_x$ [m/s]")
    # axvel.set_ylabel("$v_t$ [m/s]")
    
    # axvx.hist(np.array(init_speeds_transf)[:,0]*velocity_unit, bins = xbins, orientation='vertical')
    # plt.setp(axvx.get_xticklabels(), visible = False)
    # plt.setp(axvx.get_yticklabels(), visible = False)
    # axvt.hist(np.array(init_speeds_transf)[:,1]*velocity_unit, bins = ybins, orientation='horizontal')
    # plt.setp(axvt.get_xticklabels(), visible = False)
    # plt.setp(axvt.get_yticklabels(), visible = False)
    # # axvt.set_xlim(axvt.get_xlim()[::-1])

    # fig.tight_layout()


    # fig = plt.figure(figsize= [12, 8])
    
    # gs = gridspec.GridSpec(2, 1, figure=fig)
    
    # gs0 = gs[0].subgridspec(1,3)
    # # ax1 = fig.add_subplot(gs0[:,0])
    # ax2 = fig.add_subplot(gs0[:,0])
    # ax3 = fig.add_subplot(gs0[:,1])
    # axcappos = fig.add_subplot(gs0[:,2])
    
    # gs1 = gs[1].subgridspec(1,3)

    # axtime = fig.add_subplot(gs1[:,0])
    # # axisotope = fig.add_subplot(gs1[:,1])
    # axpos = fig.add_subplot(gs1[:,1])
    # # axvel= fig.add_subplot(gs1[:,2])
    
    # gsvel = gs1[:,2].subgridspec(6,6)
    # axvel = fig.add_subplot(gsvel[1:,:-1])
    # axvx = fig.add_subplot(gsvel[0,:-1], sharex = axvel)
    # axvt = fig.add_subplot(gsvel[1:,-1], sharey = axvel)
        
    # # fig, ax = plt.subplots(figsize = [12, 8])
    # # plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
    # # ax1.hist(init_speeds[:,0]*velocity_unit, bins = np.linspace(-4,300,305))

    # # # ax2.hist(transformed_speeds*velocity_unit, bins = np.linspace(-4,300,305), label = "Transformed final velocities")
    # # ax1.set_ylabel("Initial Counts [1]")
    # # ax1.set_xlabel("$v_x$ [m/s]")
    # # ax.legend()
    # textstr = '\n'.join([f"Runs: {MC_runtime}", f"Success: {init_speeds[:,0].size}", f"Measured: {times.size}"])
    # props = dict(boxstyle='round')
    # props2 = dict(boxstyle='round', fc = "white", alpha = 0.75)
    # props = props2
    # ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    
    # # plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
    # ax2.hist(init_speeds_no_none[:,0]*velocity_unit, bins = np.linspace(-4,300,305), label = "Initial velocities")
    # ax2.hist(transformed_speeds*velocity_unit*np.sqrt(2), bins = np.linspace(-4,300,305), label = "Final velocities")
    # # ax2.hist(transformed_speeds*velocity_unit, bins = np.linspace(-4,300,305), label = "Transformed final velocities")
    # ax2.set_ylabel("Counts [1]")
    # ax2.set_xlabel("$v_x$ [m/s]")
    # ax2.legend()
    # # textstr = '\n'.join([f"Runs: {MC_runtime}", f"Measured: {times.size}"])
    # # props = dict(boxstyle='round')
    # # ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    
    # # plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
    # ax3.hist(times*time_unit*1e3, bins = 100)
    # ax3.set_ylabel("Counts [1]")
    # ax3.set_xlabel("Time of flight [ms]")


    # # axisotope.hist(iso_samp, bins = [106,107,108,109,110,111,112,113,114,115,116,117])
    # # axisotope.set_xlabel("Isotope")
    # # axisotope.set_ylabel("Count [1]")
    
    # axtime.hist(np.array(t_samp)*time_unit*1e3, bins = 100)
    # axtime.set_xlabel("Time [ms]")
    # axtime.set_ylabel("Count [1]")
    
    
    # axpos.hist2d(*np.array(pos_samp)[:,1:].T, bins = [51, 51])
    # axpos.set_xlabel("y [cm]")
    # axpos.set_ylabel("z [cm]")
    # axpos.text(0.05, 0.95, "Start pos for sampled atoms", transform=axpos.transAxes, fontsize=14, verticalalignment='top', bbox = props2)
    
    
    # axcappos.hist2d(*np.array(capstart_poss)[:,1:].T, bins = [51, 51])
    # axcappos.set_xlabel("y [cm]")
    # axcappos.set_ylabel("z [cm]")
    # axcappos.text(0.05, 0.95, "Start pos for measured atoms", transform=axcappos.transAxes, fontsize=14, verticalalignment='top', bbox = props2)
    
    # _, xbins, ybins, _ = axvel.hist2d(*(np.array(init_speeds_transf).T*velocity_unit), bins = [51, 51])
    # axvel.set_xlabel("$v_x$ [m/s]")
    # axvel.set_ylabel("$v_y$ [m/s]")
    
    # axvx.hist(np.array(init_speeds_transf)[:,0]*velocity_unit, bins = xbins, orientation='vertical')
    # plt.setp(axvx.get_xticklabels(), visible = False)
    # plt.setp(axvx.get_yticklabels(), visible = False)
    # axvt.hist(np.array(init_speeds_transf)[:,1]*velocity_unit, bins = ybins, orientation='horizontal')
    # plt.setp(axvt.get_xticklabels(), visible = False)
    # plt.setp(axvt.get_yticklabels(), visible = False)
    # # axvt.set_xlim(axvt.get_xlim()[::-1])

    # fig.tight_layout()

# xs = np.linspace(-30,30,501)
# plt.plot(xs, [interpolate_magnet([x,0,0]) for x in xs])
# plt.plot( magnet_data[:,0]/10, magnet_data[:,1],'x')
# plt.plot(-magnet_data[:,0]/10, magnet_data[:,1],'x')
# plt.show()


# init_speed = np.linspace(100/velocity_unit, 225/velocity_unit,100)

# speeds = {}

# for i in Hamiltonians.keys():
#     speeds[i] = np.array([evolve_beam_vel(v, Hamiltonians[i], laserargs={'det_slower' : 175e6/hertz_unit}) for v in init_speed])

# # plt.subplots()
# # plt.plot(init_speed*velocity_unit, final_speed*velocity_unit, 'x')
# # plt.show()

# # from scipy.integrate import cumulative_trapezoid

# def get_density(pdf, vels, init_vels):
#     ret = []
#     for i in range(len(vels) - 1):
#         ret.append(pdf((init_vels[i] + init_vels[i+1])/2)/abs(vels[i+1] - vels[i]))
#     return np.array(ret)

# midpoint = lambda arr : np.array((arr[1:] + arr[:-1])/2)

# plt.subplots()
# plt.plot(midpoint(init_speed)*velocity_unit, get_density(capture_pdf, init_speed, init_speed))
# for i in Hamiltonians.keys():
#     mask = speeds[i] >= 0
#     plt.plot(midpoint(speeds[i])[mask[1:]]*velocity_unit, get_density(capture_pdf, speeds[i], init_speed)[mask[1:]]*abundance_data[i])
# plt.show()