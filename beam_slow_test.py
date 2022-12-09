from init import *
import random as rand

# def rej_samp(func = lambda _ : 1, rand_x = lambda : rand.uniform(0,1), rand_y = lambda x : 0):
#     while True:
#         x, y = rand_x(), rand_y()
#         if (func(x) >= y):
#             yield x

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
    return y[-3]

zero_condition.terminal = True

slower_magnet = pylcp.magField(get_interpolator([-25,0,0]))

def evolve_beam_vel(v0, ham,magnets = slower_magnet,lasers = Slow_Beam, laserargs = {'det_slower' : -175e6/hertz_unit}):
    eq = pylcp.rateeq(lasers(**laserargs),magnets, ham,include_mag_forces=False)
    try:
        eq.set_initial_pop(np.array([1., 0., 0., 0.]))
    except ValueError: # Quick and dirty solution to detect the two fermionic hamiltonians
        eq.set_initial_pop(np.array([0.5, 0.5, 0., 0., 0., 0., 0., 0.]))
    eq.set_initial_position_and_velocity([-50/cm_unit,0,0], [v0, 0, 0])
    eq.evolve_motion([0., 5e-2/time_unit], events=[zero_condition, backwards_lost], progress_bar=False, max_step = 1)
    if(eq.sol.v[0][-1] < 0):
        return -1
    return eq.sol.v[0][-1]

# rej_samp_isotope = rej_samp(func = lambda x : abundance_data[x], rand_x = rand.randint(106,116), rand_y = lambda : rand.uniform(0,1))
# rej_samp_vel = rej_samp(func = lambda x : norm.pdf(x), rand_x = lambda : rand.uniform(100,200), rand_y = lambda : rand.uniform(0,1))

# xs = np.linspace(-30,30,501)
# plt.plot(xs, [interpolate_magnet([x,0,0]) for x in xs])
# plt.plot( magnet_data[:,0]/10, magnet_data[:,1],'x')
# plt.plot(-magnet_data[:,0]/10, magnet_data[:,1],'x')
# plt.show()


init_speed = np.linspace(100/velocity_unit, 225/velocity_unit,100)

speeds = {}

for i in Hamiltonians.keys():
    speeds[i] = np.array([evolve_beam_vel(v, Hamiltonians[i], laserargs={'det_slower' : 175e6/hertz_unit}) for v in init_speed])

# plt.subplots()
# plt.plot(init_speed*velocity_unit, final_speed*velocity_unit, 'x')
# plt.show()

# from scipy.integrate import cumulative_trapezoid

def get_density(pdf, vels, init_vels):
    ret = []
    for i in range(len(vels) - 1):
        ret.append(pdf((init_vels[i] + init_vels[i+1])/2)/abs(vels[i+1] - vels[i]))
    return np.array(ret)

midpoint = lambda arr : np.array((arr[1:] + arr[:-1])/2)

plt.subplots()
plt.plot(midpoint(init_speed)*velocity_unit, get_density(capture_pdf, init_speed, init_speed))
for i in Hamiltonians.keys():
    mask = speeds[i] >= 0
    plt.plot(midpoint(speeds[i])[mask[1:]]*velocity_unit, get_density(capture_pdf, speeds[i], init_speed)[mask[1:]]*abundance_data[i])
plt.show()