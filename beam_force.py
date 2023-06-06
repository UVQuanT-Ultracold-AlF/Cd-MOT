from init import *

dr = 0.01
dv = 1/velocity_unit

r = np.arange(-25, 0+dr, 2*dr)
v = np.arange(0, 400/velocity_unit, 2*dv)

R, V = np.meshgrid(r, v)

#upper = 1_309_864.72
upper = 1_309_864.341 # GHz
laser_det = (1_309_863.55 - upper)*1e9/hertz_unit
#Rfull = np.array([sqrt(2)*R, sqrt(2)*R, np.zeros(R.shape)])
#Vfull = np.array([sqrt(2)*V, sqrt(2)*V, np.zeros(V.shape)])

pol = np.array([0,1,1j])

def Slow_Beam(det_slower, *args, pol=pol, **kwargs):
    # pol /= sum(map(lambda x : x*np.conj(x), pol))
    return pylcp.laserBeams([
        {'kvec':np.array([-1, 0., 0.]), 'pol': -1, 'delta':det_slower, 's':slower_s,'wb':slower_beam_width, 'pol_coord':'spherical'}
    ], beam_type=pylcp.gaussianBeam)


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
slower_magnet = pylcp.magField(get_interpolator([-10.5/cm_unit,0,0]))



laserargs = {'det_slower' : laser_det}
eq = pylcp.rateeq(Slow_Beam(**laserargs),slower_magnet, Hamiltonians[114],include_mag_forces=False,)
eq.generate_force_profile([R, np.zeros(R.shape), np.zeros(R.shape)],
                           [V, np.zeros(V.shape), np.zeros(V.shape)],
                           name='Frad', progress_bar=True)

plt.figure()
plt.imshow(eq.profile['Frad'].F[0], origin='lower',
           extent=(np.amin(r), np.amax(r),
                   np.amin(v*velocity_unit)-dv/2, np.amax(v*velocity_unit)-dv/2),
           aspect='auto', cmap='gnuplot')
cb1 = plt.colorbar()
plt.show()