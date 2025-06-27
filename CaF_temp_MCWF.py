import pylcp
import numpy as np
from scipy import constants as const
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import time
import pathos
import multiprocessing as mp
import dill

gamma = 2*np.pi*8.3e6 # Hz
k = 2*np.pi/606e-9 # m^-1

t_unit = 1/gamma
#t_unit = 1e-1

m_unit = 1/k
#m_unit = 1e-9

velocity_unit = m_unit/t_unit
accel_unit = m_unit/t_unit**2
Hz_unit = 1/t_unit
lin_Hz_unit = 1/t_unit/2/np.pi
Js_unit = const.hbar # kg m^2/s
mass_unit = Js_unit*t_unit/m_unit**2
HzperT_unit = const.value("Bohr magneton")/(Js_unit)
T_unit = Hz_unit/HzperT_unit
amu_unit = mass_unit/1.66e-27
cm_unit = m_unit/1e-2
F_unit = mass_unit*m_unit/t_unit**2
I_sat = (np.pi*const.h*const.c*gamma)/(3*227.5e-9**3) # W/m^2

ksim=k*m_unit
gammasim=gamma/Hz_unit

# global_det = 0.5*12*gammasim

H0_X, Bq_X, U_X, Xbasis = pylcp.hamiltonians.XFmolecules.Xstate(
    N=1, I=1/2, B=0, gamma=39.65891/lin_Hz_unit, 
    b=109.1893/lin_Hz_unit, c=40.1190/lin_Hz_unit, CI=2.876e-2/lin_Hz_unit, q0=0, q2=0,
    gS=2.0023193043622, gI=0.,
    muB=1, return_basis=True
    )
E_X = np.unique(np.diag(H0_X))

H0_A, Bq_A, Abasis = pylcp.hamiltonians.XFmolecules.Astate(
    J=1/2, I=1/2, P=+1, a=(3/2*4.8)/lin_Hz_unit, glprime=-3*.0211,
    muB=1, return_basis=True
    )
E_A = np.unique(np.diag(H0_A))

dijq = pylcp.hamiltonians.XFmolecules.dipoleXandAstates(
    Xbasis, Abasis, I=1/2, S=1/2, UX=U_X
    )

energies = np.unique(H0_X.diagonal())

hamiltonian = pylcp.hamiltonian(H0_X, H0_A, Bq_X, Bq_A, dijq, k=ksim, gamma=gammasim,mass=59/amu_unit)

# mag_field = pylcp.fields.quadrupoleMagneticField(1000*1e-4*cm_unit*HzperT_unit/Hz_unit)
mag_field = pylcp.fields.quadrupoleMagneticField(40*1e-4*cm_unit*HzperT_unit/Hz_unit)
no_mag_field = pylcp.fields.quadrupoleMagneticField(0*1e-4*cm_unit*HzperT_unit/Hz_unit)


def get_lasers(s_r, s_m, s_p, Delta, d_a, d_b, sigma_r):
    phase = [0]*6
    laserBeams = pylcp.laserBeams(
        [{'kvec':np.array([0., 0., 1.]), 'pol':-sigma_r,
          'delta':Delta - energies[0], 's':s_r, 'phase':phase[0]},
          {'kvec':np.array([0., 0., -1.]), 'pol':-sigma_r,
          'delta':Delta - energies[0], 's':s_r, 'phase':phase[1]}, 
          {'kvec':np.array([0., 1., 0.]), 'pol':sigma_r,
          'delta':Delta - energies[0], 's':s_r, 'phase':phase[0]},
          {'kvec':np.array([0., -1., 0.]), 'pol':sigma_r,
          'delta':Delta - energies[0], 's':s_r, 'phase':phase[1]},
          {'kvec':np.array([1., 0., 0.]), 'pol':sigma_r,
          'delta':Delta - energies[0], 's':s_r, 'phase':phase[0]},
          {'kvec':np.array([-1., 0., 0.]), 'pol':sigma_r,
          'delta':Delta - energies[0], 's':s_r, 'phase':phase[1]},

        {'kvec':np.array([0., 0., 1.]), 'pol':1,
          'delta':Delta - energies[-1] + d_a, 's':s_m, 'phase':phase[2]},
          {'kvec':np.array([0., 0., -1.]), 'pol':1,
          'delta':Delta - energies[-1] + d_a, 's':s_m, 'phase':phase[3]}, 
          {'kvec':np.array([1., 0., 0.]), 'pol':-1,
          'delta':Delta - energies[-1] + d_a, 's':s_m, 'phase':phase[2]},
          {'kvec':np.array([-1., 0., 0.]), 'pol':-1,
          'delta':Delta - energies[-1] + d_a, 's':s_m, 'phase':phase[3]}, 
          {'kvec':np.array([0., 1., 0.]), 'pol':-1,
          'delta':Delta - energies[-1] + d_a, 's':s_m, 'phase':phase[2]},
          {'kvec':np.array([0., -1., 0.]), 'pol':-1,
          'delta':Delta - energies[-1] + d_a, 's':s_m, 'phase':phase[3]}, 

        {'kvec':np.array([0., 0., 1.]), 'pol':-1,
          'delta':Delta - energies[-1] + d_b, 's':s_p, 'phase':phase[4]},
          {'kvec':np.array([0., 0., -1.]), 'pol':-1,
          'delta':Delta - energies[-1] + d_b, 's':s_p, 'phase':phase[5]}, 
          {'kvec':np.array([0., 1., 0.]), 'pol':1,
          'delta':Delta - energies[-1] + d_b, 's':s_p, 'phase':phase[4]},
          {'kvec':np.array([0., -1., 0.]), 'pol':1,
          'delta':Delta - energies[-1] + d_b, 's':s_p, 'phase':phase[5]}, 
          {'kvec':np.array([1., 0., 0.]), 'pol':1,
          'delta':Delta - energies[-1] + d_b, 's':s_p, 'phase':phase[4]},
          {'kvec':np.array([-1., 0., 0.]), 'pol':1,
          'delta':Delta - energies[-1] + d_b, 's':s_p, 'phase':phase[5]}, 

          ])
    
    return laserBeams

def generate_random_solution(args):
    global eqn
    tmax, rng_seed, init_v, init_r = args
    # We need to generate random numbers to prevent solutions from being seeded
    # with the same random number.
    freeze_axis = np.array([False,False,False])
    rng = np.random.default_rng(rng_seed)
    # print(np.indices(eqn.Psi0.shape))
    # print(np.cumsum(np.abs(eqn.Psi0)) > rng.uniform(0,1))
    Psi0 = np.zeros(eqn.Psi0.shape)
    Psi0[np.indices(eqn.Psi0.shape)[0,tuple(np.cumsum(np.abs(eqn.Psi0)) > rng.uniform(0,1))[0]]] = 1
    eqn.set_initial_psi(Psi0)
    eqn.set_initial_velocity(init_v*(~freeze_axis))
    eqn.set_initial_position(init_r*(~freeze_axis))
    eqn.evolve_motion([0., tmax/t_unit],
                      random_recoil=True,
                      max_scatter_probability=0.1,
                      freeze_axis=freeze_axis,
                      rtol=1e-3,
                      t_eval=np.linspace(0,tmax/t_unit, 1000),
                      atol=np.concatenate(([1e-5]*16,[1/velocity_unit,1/velocity_unit,1/velocity_unit,1e-2/cm_unit,1e-2/cm_unit,1e-2/cm_unit])),
                      method="RK23",
                      progress_bar=True,
                      rng=rng)

    return [eqn.sol.v, eqn.sol.t, eqn.sol.r, [tmax*0.999<eqn.sol.t[-1], len(eqn.sol.t_random)]]

eqn = None

def init_worker(eqn_dump):
    global eqn
    eqn = dill.loads(eqn_dump)


v_final = []
import time
if __name__ == "__main__":
    __spec__ = None
    
    # obe = pylcp.obe(MOT_Beams_nov1(-2*np.pi*43e6/Hz_unit,0,1,2*np.pi*205e6/Hz_unit), mag_field, hamiltonian,include_mag_forces=False, transform_into_re_im=True)
    # eqn = obe
    
    # eqn.set_initial_position_and_velocity(np.array([0., 0., 0.]), np.array([0., 0., 0.]))
    # if isinstance(eqn, pylcp.rateeq):
    #     eqn.set_initial_pop_from_equilibrium()
    # elif isinstance(eqn, pylcp.obe):
    #     eqn.set_initial_rho_from_rateeq()
    chunksize = 256
    N_cpus = 256
    N_atom = 256 # Needs to be divisible by chunksize   

    # if hasattr(eqn, 'sol'):
    #     del eqn.sol

    eqn = pylcp.MCWF(get_lasers(2,2,2,2*np.pi*20e6/Hz_unit,2*np.pi*1.5e6/Hz_unit,-2*np.pi*0.5e6/Hz_unit,1), mag_field, hamiltonian)
    # eqn = dill.load("AlF_nov1_-0.5.dump")
    eqn.set_initial_position_and_velocity(np.array([0., 0., 0.]), np.array([0., 0., 0.]))
    if isinstance(eqn, pylcp.rateeq):
        eqn.set_initial_pop_from_equilibrium()
    elif isinstance(eqn, pylcp.obe):
        eqn.set_initial_rho_from_rateeq()
    else:
        eqn.set_initial_psi(np.concatenate([[1/12]*12,[0]*4]))
    if hasattr(eqn, 'sol'):
        del eqn.sol
    print(eqn.recoil_velocity*velocity_unit*100," cm/s")
    print(eqn.recoil_velocity[0,12]*velocity_unit*100," cm/s")

    Rsc = gamma/50
    runtime = 15*46*1.66e-27*gamma/(2*(k**2)*Rsc*const.hbar)
    runtime = 20e-3 # 6ms

    init_vx = lambda T : np.random.normal(0, np.sqrt(1.380e-23*T/(59*1.66e-27)))
    init_rx = lambda : np.random.normal(0, 0.05/cm_unit)
    T_init = 1e-3

    # print(eqn.recoil_velocity['X(v=0)->A(v=0)']*velocity_unit*100," cm/s")
    ss = lambda : np.random.SeedSequence(time.time_ns()) # "It's the same combination as my luggage!"
    # child_seeds = ss.spawn(N_atom)
    # with pathos.pools.ProcessPool(ncpus=4) as pool:
    #     v_final += pool.map(
    #         generate_random_solution,
    #         N_atom*[eqn],
    #         N_atom*[runtime], 
    #         child_seeds
    #     )
    
    with mp.Pool(processes=N_cpus, initializer=init_worker, initargs=(dill.dumps(eqn),)) as pool:
        v_final = pool.map(generate_random_solution, [[runtime, seed, np.array([init_vx(T_init)/velocity_unit,init_vx(T_init)/velocity_unit,init_vx(T_init)/velocity_unit]),np.array([init_rx(), init_rx(), init_rx()])] for seed in ss().spawn(N_atom)])
    
    stats = np.array([f[3] for f in v_final])
    np.savez("out2.npz", v=[f[0] for f in v_final], ts=[f[1] for f in v_final], r=[f[2] for f in v_final], stats=stats)
