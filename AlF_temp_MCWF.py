import pylcp
import numpy as np
from scipy import constants as const
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
import time
import pathos
import multiprocessing as mp
import dill

gamma = 2*np.pi*84e6 # Hz
k = 2*np.pi/227.5e-9 # m^-1

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
I_sat = (np.pi*const.h*const.c*gamma)/(3*227.5e-9**3) # W/m^2

ksim=k*m_unit
gammasim=gamma/Hz_unit

global_det = 0.5*6*gammasim

labels = [(3/2,1),(3/2,2),(5/2,2),(5/2,3),(7/2,3),(7/2,4)]
full_labels = np.concatenate([[(i[0], i[1], j) for j in np.arange(-i[1],i[1]+1,1)] for i in labels])

def full_red(j):
    return np.sqrt(2*j+1)

def AState_mu_q(J, lbls, I1, I2): # for the A1Pi state
    Lambda = 1
    def matrix_element(p,F1,F,MF,F1p,Fp,MFp):
        return -float((-1)**(F-MF)*wigner_3j(F,1,Fp,-MF,p,MFp)*\
            np.sqrt(3*(2*F+1)*(2*Fp+1))*wigner_9j(F,Fp,1,F1,F1p,1,I2,I2,0)*full_red(I2)*\
            np.sqrt(3*(2*F1+1)*(2*F1p+1))*wigner_9j(F1,F1p,1,J,J,1,I1,I1,0)*full_red(I1)*\
            Lambda*(-1)**(J-Lambda)*np.sqrt((2*J+1)*(2*J+1))*wigner_3j(J,1,J,-Lambda,0,Lambda))
    return np.array([[[matrix_element(i,*l1,*l2) for l2 in lbls] for l1 in lbls] for i in [-1,0,1]])

def Q_d_q(J, exlabels, grlabels, I1, I2):
    def matrix_element(p,F1,F,MF,F1p,Fp,MFp):
        return float((-1)**(F-MF)*wigner_3j(F,1,Fp,-MF,p,MFp)*np.sqrt(3*(2*F+1)*(2*Fp+1))*wigner_9j(F,Fp,1,F1,F1p,1,I2,I2,0)*np.sqrt(3*(2*F1+1)*(2*F1p+1))*wigner_9j(F1,F1p,1,J,J,1,I1,I1,0))*full_red(I2)*full_red(I1)
    raw_dq =  np.array([[[matrix_element(i,*l1,*l2) for l2 in exlabels] for l1 in grlabels] for i in [-1,0,1]])
    norm = np.sqrt(np.sum(np.abs(raw_dq)**2,axis=(0,1)))
    return np.einsum("ijk,k->ijk",raw_dq,1/norm)

def H0_Singlet(J, Lambda, I1, I2, ai = [0,0], eq0Q = 0, eq2Q = 0, DI = 0): # DOES NOT calculate cross terms between J!!!
    def _delta(i, j, eps = 1e-6):
        return abs(i-j) < eps
    def nuclear_rotation(F1, F, MF, F1p, Fp, MFp):
        I2dotL = (-1)**(2*F1p + F + I2 + 1 + I1 - Lambda)*_delta(MF,MFp)*_delta(F,Fp)\
            *wigner_6j(I2,F1p,F,F1,I2,1)*wigner_6j(J,F1p,I1,F1,J,1)*wigner_3j(J,1,J,-Lambda,0,Lambda)\
            *np.sqrt((2*F1+1)*(2*F1p+1)*(2*J+1)*(2*J+1)*I2*(I2+1)*(2*I2+1))*Lambda
        I1dotL = (-1)**(J+F1+I1+J-Lambda)*_delta(MF,MFp)*_delta(F,Fp)*_delta(F1,F1p)\
            *wigner_6j(I1,J,F1,J,I1,1)\
            *wigner_3j(J,1,J,-Lambda,0,Lambda)*np.sqrt((2*J+1)*(2*J+1)*I1*(I1+1)*(2*I1+1))*Lambda
        return float(ai[0]*I1dotL + ai[1]*I2dotL)
    
    def quadrupole(F1, F, MF, F1p, Fp, MFp):
        return float((-1)**(J+F1+I1+J-Lambda)*_delta(MF,MFp)*_delta(F,Fp)*_delta(F1,F1p)\
            *(wigner_6j(I1,J,F1,J,I1,2)/wigner_3j(I1,2,I1,-I1,0,I1))*np.sqrt((2*J+1)*(2*J+1))\
            *(wigner_3j(J,2,J,-Lambda,0,Lambda)*(eq0Q/4)\
            +np.sum([wigner_3j(J,2,J,-Lambda,q,Lambda)*(eq2Q/(4*np.sqrt(6)))*_delta(Lambda,Lambda+q) for q in [-2,2]])))

    def nucleon_nucleon(F1, F, MF, F1p, Fp, MFp):
        return float(-DI*np.sqrt(30)*(-1)**(F1p+F+I2)*_delta(MF,MFp)*_delta(F,Fp)*wigner_6j(I2,F1p,F,F1,I2,1)\
            *np.sqrt((2*F1+1)*(2*F1p+1))*wigner_9j(F1,F1p,1,J,J,2,I1,I1,1)*(-1)**(J-Lambda)\
            *wigner_3j(J,2,J,-Lambda,0,Lambda)*np.sqrt((2*J+1)*(2*J+1)*I1*(I1+1)*(2*I1+1)*I2*(I2+1)*(2*I2+1)))
        
    I12 = int(2*I1)
    I22 = int(2*I2)
    J2 = int(2*J)
    state_labels = np.array([(F12/2,F2/2,MF2/2) for F12 in range(abs(J2-I12),J2+I12+1,2) for F2 in range(abs(F12-I22),F12+I22+1,2) for MF2 in range(-F2,F2+1,2)])
    H0 = np.zeros((state_labels.shape[0],state_labels.shape[0]))
    H0 += np.array([[nuclear_rotation(*labket,*labbra) for labket in state_labels] for labbra in state_labels])
    H0 += np.array([[quadrupole(*labket,*labbra) for labket in state_labels] for labbra in state_labels])
    H0 += np.array([[nucleon_nucleon(*labket,*labbra) for labket in state_labels] for labbra in state_labels])
    return state_labels, H0

labX, H0X = H0_Singlet(1,0,5/2,1/2,eq0Q=-37.526,DI=0.0066)
# H0X -= np.min(H0X)

labA, H0A = H0_Singlet(1,1,5/2,1/2,ai=[113,181])
# H0A -= np.min(H0A)

mu_q = {}
d_q = {}
H0 = {}

mu_q['X(v=0)'] = np.zeros((3,full_labels.shape[0],full_labels.shape[0]))
# mu_q['X(v=1)'] = np.zeros((3,full_labels.shape[0],full_labels.shape[0]))
# mu_q['X(v=2)'] = np.zeros((3,full_labels.shape[0],full_labels.shape[0]))
# H0['X(v=0)'] = (2*np.pi/Hz_unit)*np.diag(np.concatenate([[11.235e6]*8, [0]*12, [7.914e6]*16]))
# H0['X(v=1)'] = (2*np.pi/Hz_unit)*np.diag(np.concatenate([[11.235e6]*8, [0]*12, [7.914e6]*16]))
# H0['X(v=1)'] = np.zeros((full_labels.shape[0],full_labels.shape[0]))
# H0['A(v=0)'] = (2*np.pi/Hz_unit)*np.diag([9.06039e7, 9.06039e7, 9.06039e7, 0., 0., 0., 0., 0., 2.09987e8, 2.09987e8, 2.09987e8, 2.09987e8, 2.09987e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8])
H0['X(v=0)'] = H0X*1e6*2*np.pi/Hz_unit
# H0['X(v=1)'] = H0X*1e6*2*np.pi/Hz_unit
# H0['X(v=2)'] = H0X*1e6*2*np.pi/Hz_unit
H0['A(v=0)'] = H0A*1e6*2*np.pi/Hz_unit - global_det*np.eye(H0A.shape[0])
# H0['A(v=1)'] = H0A*1e6*2*np.pi/Hz_unit

# mu_q['X(v=0)'] = np.zeros((3,full_labels.shape[0],full_labels.shape[0]))
# #mu_q['X(v=1)'] = np.zeros((3,full_labels.shape[0],full_labels.shape[0]))
# H0['X(v=0)'] = (2*np.pi/Hz_unit)*np.diag(np.concatenate([[11.235e6]*8, [0]*12, [7.914e6]*16]))
# #H0['X(v=1)'] = np.zeros((full_labels.shape[0],full_labels.shape[0]))
# H0['A(v=0)'] = (2*np.pi/Hz_unit)*np.diag([9.06039e7, 9.06039e7, 9.06039e7, 0., 0., 0., 0., 0., 2.09987e8, 2.09987e8, 2.09987e8, 2.09987e8, 2.09987e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 1.99498e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 3.61039e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8, 4.36374e8])

mu_q['A(v=0)'] = AState_mu_q(1, full_labels, 5/2,1/2)

dq = Q_d_q(1,labA,labX,5/2,1/2)

# dijq = np.einsum("ijk->kij",dijq)
d_q[("X(v=0)","A(v=0)")] = dq#*np.sqrt(0.99)

hamiltonian = pylcp.hamiltonian(mass=46/amu_unit, k=ksim, gamma=gammasim)
[hamiltonian.add_H_0_block(l, H) for l, H in H0.items()]
[hamiltonian.add_mu_q_block(l, mu, muB=1) for l, mu in mu_q.items()]
[hamiltonian.add_d_q_block(l[0],l[1], dq, k=ksim, gamma=gammasim) for l, dq in d_q.items()]
# hamiltonian.add_d_q_block("X(v=0)","A(v=0)",dijq,k=0,gamma=gammasim*0.99)
# hamiltonian.add_d_q_block("A(v=0)","X(v=1)",dijq,k=0,gamma=gammasim*0.01)
# hamiltonian.print_structure()

mag_field = pylcp.fields.quadrupoleMagneticField(1000*1e-4*cm_unit*HzperT_unit/Hz_unit)
no_mag_field = pylcp.fields.quadrupoleMagneticField(0*1e-4*cm_unit*HzperT_unit/Hz_unit)

s = 0.1

def MOT_Beams_nov1(det_MOT, p, *args):
    return {
    'X(v=0)->A(v=0)' : pylcp.laserBeams([
        {'kvec':np.array([1,0, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s},
        {'kvec':np.array([-1,0, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s},
        {'kvec':np.array([0, -1, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s},
        {'kvec':np.array([0, 1, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s},
        {'kvec':np.array([0., 0.,  1.])*ksim, 'pol':-p, 'delta':det_MOT, 's':s},
        {'kvec':np.array([0., 0., -1.])*ksim, 'pol':-p, 'delta':det_MOT, 's':s},
    ], beam_type=pylcp.infinitePlaneWaveBeam)}
    
# def MOT_Beams_nov1(det_MOT, p, *args):
#     return {
#     'X(v=0)->A(v=0)' : pylcp.laserBeams([
#         {'kvec':np.array([1,0, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s,'wb':0.3*2/cm_unit},
#         {'kvec':np.array([-1,0, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s,'wb':0.3*2/cm_unit},
#         {'kvec':np.array([0, -1, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s,'wb':0.3*2/cm_unit},
#         {'kvec':np.array([0, 1, 0.])*ksim, 'pol':p, 'delta':det_MOT, 's':s,'wb':0.3*2/cm_unit},
#         {'kvec':np.array([0., 0.,  1.])*ksim, 'pol':-p, 'delta':det_MOT, 's':s,'wb':0.3*2/cm_unit},
#         {'kvec':np.array([0., 0., -1.])*ksim, 'pol':-p, 'delta':det_MOT, 's':s,'wb':0.3*2/cm_unit},
#     ], beam_type=pylcp.gaussianBeam)}
    

#eqn = pylcp.rateeq(MOT_Beams_nov1(-2*np.pi*43e6/Hz_unit,0,1,2*np.pi*205e6/Hz_unit), mag_field, hamiltonian,include_mag_forces=False)
#num_of_scatters = np.zeros((N_atom,), dtype='int')
#num_of_steps = np.zeros((N_atom,), dtype='int')

def generate_random_solution(args):
    global eqn
    tmax, rng_seed, init_v, init_r = args
    # We need to generate random numbers to prevent solutions from being seeded
    # with the same random number.
    freeze_axis = np.array([False,False,False])
    eqn.set_initial_velocity(init_v*(~freeze_axis))
    eqn.set_initial_position(init_r*(~freeze_axis))
    eqn.evolve_motion([0., tmax/t_unit],
                      random_recoil=True,
                      max_scatter_probability=0.1,
                      freeze_axis=freeze_axis,
                      rtol=1e-3,
                      t_eval=np.linspace(0,tmax/t_unit, 1000),
                      atol=np.concatenate(([1e-5]*72,[1/velocity_unit,1/velocity_unit,1/velocity_unit,1e-2/cm_unit,1e-2/cm_unit,1e-2/cm_unit])),
                      method="RK23",
                      progress_bar=True,
                      rng=np.random.default_rng(rng_seed))

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

    eqn = pylcp.MCWF(MOT_Beams_nov1(6*gammasim - global_det,p=1), mag_field, hamiltonian, a = np.array([0,0,-0*10./accel_unit]))
    # eqn = dill.load("AlF_nov1_-0.5.dump")
    eqn.set_initial_position_and_velocity(np.array([0., 0., 0.]), np.array([0., 0., 0.]))
    if isinstance(eqn, pylcp.rateeq):
        eqn.set_initial_pop_from_equilibrium()
    elif isinstance(eqn, pylcp.obe):
        eqn.set_initial_rho_from_rateeq()
    else:
        eqn.set_initial_psi(np.concatenate([[1/36]*36,[0]*36]))
    if hasattr(eqn, 'sol'):
        del eqn.sol
    print(eqn.recoil_velocity*velocity_unit*100," cm/s")
    print(eqn.recoil_velocity[0,36]*velocity_unit*100," cm/s")

    Rsc = gamma/50
    runtime = 15*46*1.66e-27*gamma/(2*(k**2)*Rsc*const.hbar)
    runtime = 6e-3 # 6ms

    init_vx = lambda T : np.random.normal(0, np.sqrt(1.380e-23*T/(46*1.66e-27)))
    init_rx = lambda : np.random.normal(0, 0.0125/cm_unit)
    T_init = 0e-3

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
