from init import *
import random as rand

def rej_samp(func = lambda _ : 1, rand_x = lambda : rand.uniform(0,1), rand_y = lambda : 0):
    while True:
        x, y = rand_x(), rand_y()
        # print (x, y, func(x))
        if (func(x) >= y):
            yield x

rej_samp_isotope = rej_samp(func = lambda x : abundance_data[x], rand_x = rand.randint(106,116), rand_y = lambda : rand.uniform(0,1))
rej_samp_vel = rej_samp(func = lambda x : capture_pdf(x), rand_x = lambda : rand.uniform(100/velocity_unit,200/velocity_unit), rand_y = lambda : rand.uniform(0,1))
rej_samp_angle = rej_samp(rand_x = lambda : rand.uniform(10e-3,10e-3))
rej_samp_time = rej_samp(rand_x = lambda : rand.uniform(0,1e-3/time_unit))
rej_samp_pos = rej_samp(func = lambda r : 1 if sum(map(lambda x : x**2, r)) < (1.5e-3)**2 else 0, rand_x = lambda : (rand.uniform(-1.5e-3,1.5e-3),rand.uniform(-1.5e-3,1.5e-3)), rand_y = lambda : 0.5)

magnet_data = np.loadtxt("./csv/RingMagnet_BzProfile.csv",delimiter="\t")
def get_interpolator(pos):
    def interpolate_magnet(R):
        nonlocal pos
        R = R - pos
        x = abs(R[0])*10
        prevpos = None
        prevB = None
        B = None
        for pos, Bz in magnet_data:
            if(x < pos):
                break
            prevpos = pos
            prevB = Bz
        else:
            B = 0
        if (B is None):
            B = prevB + (x - prevpos)*(Bz - prevB)/(pos - prevpos)
        
        B = B*10**(-4)*consts.value('Bohr magneton')/(10**6*91*consts.h)

        return [B,0,0]
    return interpolate_magnet

def Slow_Beam(det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1, 0., 0.]), 'pol':-1, 'delta':det_slower, 's':slower_s,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def zero_condition(t,y):
    return sum(map(lambda x : x**2, y[-3:])) - 1

def lost_forwards(t, y):
    return y[-3] - 2

lost_forwards.terminal = True
zero_condition.terminal = True

slower_magnet = pylcp.magField(get_interpolator([-10.5/cm_unit,0,0]))

def evolve_beam_vel(v0, ham,magnets = slower_magnet,lasers = Slow_Beam, laserargs = {'det_slower' : -175e6/hertz_unit}, angle = 0, time = 0, r = [0,0]):
    eq = pylcp.rateeq(lasers(**laserargs),magnets, ham,include_mag_forces=False)
    try:
        eq.set_initial_pop(np.array([1., 0., 0., 0.]))
    except ValueError: # Quick and dirty solution to detect the two fermionic hamiltonians
        eq.set_initial_pop(np.array([0.5, 0.5, 0., 0., 0., 0., 0., 0.]))
    eq.set_initial_position_and_velocity([-45.5/cm_unit,r[0],r[1]], [v0*np.cos(angle), v0*np.sin(angle), 0])
    eq.evolve_motion([time, time + 1/time_unit], events=[zero_condition, backwards_lost], progress_bar=False, max_step = 1)
    if(eq.sol.v[0][-1] < 0):
        return -1,-1
    
    if eq.sol.r[0][-1] > 2:
        return -2,-2
    return eq.sol.v[0][-1], eq.sol.t[-1]

MC_runtime = int(1e5)

speeds = []
times = []
init_speeds = []
for i in range(MC_runtime):
    print(f"{i+1}/{MC_runtime}: {100*(i+1)/MC_runtime:.2f}%", end = '\r')
    v0 = next(rej_samp_vel)
    init_speeds.append(v0)
    speed, time = evolve_beam_vel(v0, Hamiltonians[114], laserargs={'det_slower' : -607.5e6/hertz_unit}, angle = next(rej_samp_angle), time = next(rej_samp_time), r = next(rej_samp_pos))
    if (speed < 0):
        continue
    speeds.append(speed)
    times.append(time)

speeds = np.asarray(speeds)
init_speeds = np.asarray(init_speeds)
times = np.asarray(times)

plt.figure(figsize = [12,8])
# plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
plt.hist(init_speeds*velocity_unit, bins = np.linspace(-4,300,305), label = "Initial velocities")
plt.hist(speeds*velocity_unit, bins = np.linspace(-4,300,305), label = "Final velocities")
plt.ylabel("Counts [1]")
plt.xlabel("$v_x$ [m/s]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize = [12,8])
# plt.plot(speeds*velocity_unit, np.linspace(0,10,speeds.size) , "x")
plt.hist(times*time_unit*1e3, bins = 100)
plt.ylabel("Counts [1]")
plt.xlabel("Time of flight [ms]")
plt.tight_layout()
plt.show()


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