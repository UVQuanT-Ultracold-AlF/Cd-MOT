import pylcp
import numpy as np
from scipy import constants as const
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import time
import pathos
import multiprocessing as mp
import dill

atom = pylcp.atom('87Rb')
# gamma = 2*np.pi*5.75e6 # Hz
# k = 2*np.pi/794.97851156 # m^-1
gamma = 2*np.pi*atom.state[1].gammaHz # Hz
k = 2*np.pi/794.97851156e-9 # m^-1

t_unit = 1/gamma
#t_unit = 1e-1

m_unit = 1/k
#m_unit = 1e-9

velocity_unit = m_unit/t_unit
accel_unit = m_unit/t_unit**2
Hz_unit = 1/t_unit
Js_unit = const.hbar # kg m^2/s
mass_unit = Js_unit*t_unit/m_unit**2
HzperT_unit = const.value("Bohr magneton")/(Js_unit)
T_unit = Hz_unit/HzperT_unit
amu_unit = mass_unit/1.66e-27
cm_unit = m_unit/1e-2
F_unit = mass_unit*m_unit/t_unit**2
I_sat = (np.pi*const.h*const.c*gamma)/(3*(2*np.pi/k)**3) # W/m^2

ksim=k*m_unit
gammasim=gamma/Hz_unit

Hg, mu_qg = pylcp.hamiltonians.hyperfine_coupled(
    atom.state[0].J, atom.I, atom.state[0].gJ, atom.gI,
    atom.state[0].Ahfs*2*np.pi/Hz_unit, Bhfs=0, Chfs=0,
    muB=1)
He, mu_qe = pylcp.hamiltonians.hyperfine_coupled(
    atom.state[1].J, atom.I, atom.state[1].gJ, atom.gI,
    atom.state[1].Ahfs*2*np.pi/Hz_unit, Bhfs=0, Chfs=0,
    muB=1)
dijq = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(
    atom.state[0].J, atom.state[1].J, atom.I)

Ee = np.unique(np.diagonal(He))*Hz_unit/2/np.pi
Eg = np.unique(np.diagonal(Hg))*Hz_unit/2/np.pi

mag_field = pylcp.fields.quadrupoleMagneticField(110*1e-4*cm_unit*HzperT_unit/Hz_unit)
no_mag_field = pylcp.fields.quadrupoleMagneticField(0*1e-4*cm_unit*HzperT_unit/Hz_unit)

hamiltonian = pylcp.hamiltonian(Hg, He, mu_qg, mu_qe, dijq, mass = 87/amu_unit, k = ksim, gamma = gammasim)

def MOT_Beams(delta_1, delta_2, *args):

    lasers  = pylcp.laserBeams()
    lasers += pylcp.laserBeams([
        {'kvec':np.array([-1, 0, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[1]+Ee[1]+delta_1)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([1, 0, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[1]+Ee[1]+delta_1)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0, -1, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[1]+Ee[1]+delta_1)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0, 1, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[1]+Ee[1]+delta_1)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0., 0.,  1.])*ksim, 'pol':-1, 'delta':2*np.pi*(-Eg[1]+Ee[1]+delta_1)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0., 0., -1.])*ksim, 'pol':-1, 'delta':2*np.pi*(-Eg[1]+Ee[1]+delta_1)/Hz_unit, 's':1, 'wb':1/cm_unit}
    ], beam_type=pylcp.gaussianBeam)
    lasers +=  pylcp.laserBeams([
        {'kvec':np.array([-1, 0, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[0]+Ee[1]+delta_1+delta_2)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([1, 0, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[0]+Ee[1]+delta_1+delta_2)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0, -1, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[0]+Ee[1]+delta_1+delta_2)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0, 1, 0.])*ksim, 'pol':1, 'delta':2*np.pi*(-Eg[0]+Ee[1]+delta_1+delta_2)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0., 0.,  1.])*ksim, 'pol':-1, 'delta':2*np.pi*(-Eg[0]+Ee[1]+delta_1+delta_2)/Hz_unit, 's':1, 'wb':1/cm_unit},
        {'kvec':np.array([0., 0., -1.])*ksim, 'pol':-1, 'delta':2*np.pi*(-Eg[0]+Ee[1]+delta_1+delta_2)/Hz_unit, 's':1, 'wb':1/cm_unit}
    ], beam_type=pylcp.gaussianBeam)
    return lasers

    


def generate_random_solution(args):
    global eqn
    tmax, rng_seed, v = args
    # We need to generate random numbers to prevent solutions from being seeded
    # with the same random number.
    eqn.set_initial_velocity(np.array(v))
    ts = np.linspace(0, tmax/t_unit, 1000)
    eqn.evolve_motion([0., tmax/t_unit],
                      max_step=1e-2/t_unit,
                      random_recoil=True,
                      max_scatter_probability=0.25,
                      freeze_axis=[False, False, False],
                      t_eval=ts,
                      rtol=1e-7,
                      atol=np.concatenate(([1e-5]*16*16,[1e-3/velocity_unit,1e-3/velocity_unit,1e-3/velocity_unit,1e-2/cm_unit,1e-2/cm_unit,1e-2/cm_unit])),
                      method="RK23",
                      progress_bar=True,
                      rng=np.random.default_rng(rng_seed))


    return [eqn.sol.v, eqn.sol.t, eqn.sol.r, len(eqn.sol.t_random)]

def init_worker(eqn_dump):
    global eqn
    eqn = dill.loads(eqn_dump)
    
if __name__ == "__main__":
    __spec__ = None
    import sys
    obe = pylcp.obe(MOT_Beams(150e6, float(sys.argv[1])*1e6), mag_field, hamiltonian,include_mag_forces=False, transform_into_re_im=False)

    from pylcp.common import progressBar
    eqn = obe
    #eqn = pylcp.rateeq(MOT_Beams_nov1(-2*np.pi*43e6/Hz_unit,0,1,2*np.pi*205e6/Hz_unit), mag_field, hamiltonian,include_mag_forces=False)
    N_atom = 1
    N_cpus = 1
    chunksize = 256
    #num_of_scatters = np.zeros((N_atom,), dtype='int')
    #num_of_steps = np.zeros((N_atom,), dtype='int')

    eqn.set_initial_position_and_velocity(np.array([0., 0., 0.]), np.array([0., 0., 0.]))
    eqn.set_initial_rho_from_rateeq()
    if isinstance(eqn, pylcp.rateeq):
        eqn.set_initial_pop_from_equilibrium()
    elif isinstance(eqn, pylcp.obe):
        eqn.set_initial_rho_from_rateeq()
    if hasattr(eqn, 'sol'):
        del eqn.sol
      
    print(np.array(list(obe.recoil_velocity.values()))*velocity_unit*100,"cm/s")
    Rsc = gamma/2
    runtime = 4e-8#10*87*1.66e-27*gamma/(2*(k**2)*Rsc*const.hbar)

    init_vx = lambda T : np.random.normal(0, np.sqrt(1.380e-23*T/(87*1.66e-27)))
    T_init = 1e-6

    import time
    v_final = []
    ss = lambda : np.random.SeedSequence(time.time_ns()) # "It's the same combination as my luggage!"

    # for jj in range(int(N_atom/chunksize)):
    #     with pathos.pools.ProcessPool(ncpus=256) as pool:
    #         v_final += pool.map(
    #             generate_random_solution,
    #             chunksize*[eqn],
    #             chunksize*[runtime],
    #             child_seeds[jj*chunksize:(jj+1)*chunksize]
    #         )

    with mp.Pool(processes=N_cpus, initializer=init_worker, initargs=(dill.dumps(eqn),)) as pool:
        v_final = pool.map(generate_random_solution, [[runtime, seed, [init_vx(T_init)/velocity_unit,init_vx(T_init)/velocity_unit,init_vx(T_init)/velocity_unit]] for seed in ss().spawn(N_atom)])

    stats = np.array([f[3] for f in v_final])
    np.savez("out.npz", v=[f[0] for f in v_final], r=[f[2] for f in v_final], ts=[f[1] for f in v_final], stats=stats)

