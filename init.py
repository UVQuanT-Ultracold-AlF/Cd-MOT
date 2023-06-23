import ipynb
import numpy as np
import matplotlib.pyplot as plt
import pylcp
import scipy.constants as consts
import lmfit
from pylcp.common import progressBar
import ast
import csv
import pickle
from scipy.optimize import bisect
from tqdm import tqdm
from ipynb.fs.full.MagnetsInterpolation import ComsolMagCylindricalInterpolation as mi
import inspect
from matplotlib.ticker import MultipleLocator

plt.ion()

loadpath = "./csv/"
savepath = "./images/"

# #laser parameters
# laser_det = -2
# ham_det = 0
# #laser power in Watts
# laser_power = 0.1
# #beam intensity 1/e^2 radius in cm
# beamWidth = 0.4
# #saturation intensity in W/cm^2
# Isat = 1
# s = laser_power/(Isat*np.pi*beamWidth**2)

import matplotlib as mpl

rcparams = {
# 'axes.titlesize'    : 18,
# 'axes.labelsize'    : 16,
# 'xtick.labelsize'   : 12,
# 'ytick.labelsize'   : 12,
# 'legend.fontsize'   : 12,
'font.size'         : 20
}
for e in rcparams.keys():
    mpl.rcParams[e] = rcparams[e]

k = 4369238.4 # m^-1

hbar = consts.hbar
h = consts.h

hertz_unit = 91e6
# time_unit = 1/hertz_unit
time_unit = k*1e-2/hertz_unit # Needed due to the two ways units are measured

amu_const = consts.value('atomic mass constant')

#scaled mass as described in the examples, for 114 Cd
scaledMass = 4.87e-3
# print(scaledMass)
scaledMass = 113.9*amu_const*91*(2*np.pi)/(hbar*2*np.pi*k**2)
# print(scaledMass)
# amu_unit = 113.90336500/scaledMass
amu_unit = 1/(amu_const*91*(2*np.pi)/(hbar*2*np.pi*k**2))
velocity_unit = hertz_unit/k


# laser parameters from Simon
slower_beam_width = 0.5/2 # cm
slower_I = 0.04 #0.3 # W/cm^2
slower_detuning = 0 # Placeholder
MOT_detuning = -1.45*100e6/hertz_unit
MOT_beam_width = 0.4/2 # cm
Isat = 1.1
slower_s = 2*slower_I/(np.pi*(slower_beam_width**2))/Isat
# slower_s = 0.2
# slower_s = 0.4
# slower_s = 0.3
# slower_s = 0.02
MOT_s = 1.5/Isat
cm_unit = 1


# read in initial vel_dist
vel_dist_raw_data = np.loadtxt("./csv/ChirpedSlowing_112_initialDistribution.dat")
vel_dist_raw_data = vel_dist_raw_data[vel_dist_raw_data[:,1].argsort()]

vel_dist_data = np.array([[i/velocity_unit,np.sum(vel_dist_raw_data[:,2], where=vel_dist_raw_data[:,1] == i)] for i in np.unique(vel_dist_raw_data[:,1])])
vel_dist = lambda x : np.interp(x, *(vel_dist_data.T))

# Some parameters for the MOT
# Mass taken from IAEA
mass = {106 : 105.9064598/amu_unit, 108 : 107.9041836/amu_unit, 110 : 109.9030075/amu_unit, 111 : 110.9041838/amu_unit, 112 : 111.90276390/amu_unit, 113 : 112.90440811/amu_unit, 114 : 113.90336500/amu_unit, 116 : 115.90476323/amu_unit}
abundance_data = {106 : 0.0125, 108 : 0.0089, 110 : 0.1249, 111 : 0.1280, 112 : 0.2413, 113 : 0.1222, 114 : 0.2873, 116 : 0.0749}

# Take isotope shift data from here: https://arxiv.org/pdf/2210.11425.pdf [Table II, this work]
isotope_shifts = {106 : 1818.1e6/hertz_unit, 108 : 1336.5e6/hertz_unit, 110 : 865e6/hertz_unit, 111 : 805.0e6/hertz_unit, 112 : 407.5e6/hertz_unit, 113 : 344.9e6/hertz_unit, 114 : 0e6/hertz_unit, 116 : -316.1e6/hertz_unit}
isotope_shift_hyperfine = {111 : (899.2e6 - 616.5e6)/hertz_unit, 113 : (443.4e6 - 147.8e6)/hertz_unit}

# Hamiltonians
ham_det = 0

def gen_Boson_Hamiltonian(isotope = 114):
    Hg, Bgq = pylcp.hamiltonians.singleF(F=0, gF=0, muB=1)
    He, Beq = pylcp.hamiltonians.singleF(F=1, gF=1, muB=1)
    dijq = pylcp.hamiltonians.dqij_two_bare_hyperfine(0, 1)
    return pylcp.hamiltonian(Hg, He + (isotope_shifts[isotope] - ham_det)*np.eye(3), Bgq, Beq, dijq,mass=mass[isotope],k=1, gamma=1,muB=1)

def gen_Fermion_Hamiltonian(isotope = 111):
    #Hamiltonian for F=1/2 -> F = 1/2,3/2 
    HgFermion, BgqFermion = pylcp.hamiltonians.hyperfine_coupled(J=0, I=1/2, gJ=0, gI=0, Ahfs=0, Bhfs=0, Chfs=0, muB=1)
    # HeFermion, BeqFermion = pylcp.hamiltonians.hyperfine_coupled(J=1, I=1/2, gJ=1, gI=0, Ahfs=2.2, Bhfs=0, Chfs=0, muB=1)
    Ahfs = 2*isotope_shift_hyperfine[isotope]/3
    HeFermion, BeqFermion = pylcp.hamiltonians.hyperfine_coupled(J=1, I=1/2, gJ=1, gI=0, Ahfs=Ahfs, Bhfs=0, Chfs=0, muB=1)
    dijqFermion = pylcp.hamiltonians.dqij_two_hyperfine_manifolds(J=0, Jp=1, I=0.5)
    return pylcp.hamiltonian(HgFermion, HeFermion + (isotope_shifts[isotope] - ham_det)*np.eye(6), BgqFermion, BeqFermion, dijqFermion,mass=mass[isotope],k=1, gamma=1,muB=1)

Hamiltonians = {}
# Generate all Hamiltonians
for key in mass.keys():
    if (key in isotope_shift_hyperfine.keys()):
        Hamiltonians[key] = gen_Fermion_Hamiltonian(key)
        continue
    Hamiltonians[key] = gen_Boson_Hamiltonian(key)


# Laser fields
def MOT_and_Slow_Beams(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':-1, 'delta':det_slower, 's':slower_s,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_and_Slow_Beams_timed(det_MOT, det_slower, t_cutoff, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':-1, 'delta':det_slower, 's': lambda t : slower_s if 0 > t - t_cutoff and t-t_cutoff > -0.5e-3/time_unit else 0,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_and_Slow_Beams_sig_2(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':+1, 'delta':0*slower_detuning + det_slower, 's':slower_s,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_and_Slow_Beams_lin(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':np.array([0., 1., 0.]), 'pol_coord':'cartesian', 'delta':0*slower_detuning + det_slower, 's':slower_s,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_Beams(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_Beams_infinite(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([1., 0., 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s},#,'wb':MOT_beam_width},
        {'kvec':np.array([-1., 0., 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s},#,'wb':MOT_beam_width},
        {'kvec':np.array([0., 1., 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s},#,'wb':MOT_beam_width},
        {'kvec':np.array([0., -1., 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s},#,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s},#,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s},#,'wb':MOT_beam_width}
    ], beam_type=pylcp.infinitePlaneWaveBeam)

time_range = 10e-3/time_unit

def MOT_and_Slow_Beams_timed2(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':-1, 'delta':det_slower, 's': lambda t : slower_s if t < time_range else 0,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_and_Slow_Beams_sig_2_timed(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':+1, 'delta':0*slower_detuning + det_slower, 's': lambda t : slower_s if t < time_range else 0,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_and_Slow_Beams_lin_timed(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':np.array([0., 1., 0.]), 'pol_coord':'cartesian', 'delta':0*slower_detuning + det_slower, 's': lambda t : slower_s if t < time_range else 0,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

ramp_cutoff = 10e-3/time_unit

def MOT_and_Slow_Beams_timed3(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':-1, 'delta':det_slower, 's': lambda t : slower_s*(1 if t > ramp_cutoff else t/ramp_cutoff) if t < time_range else 0,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_and_Slow_Beams_sig_2_timed2(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':+1, 'delta':0*slower_detuning + det_slower, 's': lambda t : slower_s*(1 if t > ramp_cutoff else t/ramp_cutoff) if t < time_range else 0,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_and_Slow_Beams_lin_timed2(det_MOT, det_slower, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':0*MOT_detuning + det_MOT, 's':MOT_s,'wb':MOT_beam_width},
        {'kvec':np.array([-1, 0., 0.]), 'pol':np.array([0., 1., 0.]), 'pol_coord':'cartesian', 'delta':0*slower_detuning + det_slower, 's': lambda t : slower_s*(1 if t > ramp_cutoff else t/ramp_cutoff) if t < time_range else 0,'wb':slower_beam_width}
    ], beam_type=pylcp.gaussianBeam)

permMagnets=mi('./csv/2D_Br_updated_30mmbore.csv', './csv/2D_Bz_updated_30mmbore.csv',91,-1)
permMagnetsPylcp = pylcp.magField(permMagnets.fieldCartesian)

def mod_slower_s(new_slower_s):
    global slower_s
    slower_s = new_slower_s

# def captured_condition(t, y):
#     return (y[-6]**2 + y[-3]**2) - 1e-2

def captured_condition(t, y):
    if sum(map(lambda x : x**2, y[-3:])) > 1e-4:
        return -1
    if sum(map(lambda x : (velocity_unit*x)**2, y[-6:-3])) > 0.1:
        return -1
    return 1

def weak_cap_cond(v,p):
    if sum(map(lambda x : x**2, p)) > 2e-4:
        return -1
    if sum(map(lambda x : (velocity_unit*x)**2, v)) > 0.2:
        return -1
    return 1

def lost_condition(t, y):
    return y[-3]- 8

def backwards_lost(t, y):
    if (y[-3] < -10 and y[-6]*velocity_unit < 0):
        return -1
    return 1

captured_condition.terminal = True
lost_condition.terminal = True
backwards_lost.terminal = True

m_step = 1e-4/time_unit

def captureVelocityForEq(det_MOT, det_slower, ham, lasers = MOT_and_Slow_Beams):
    print (f"{det_MOT*hertz_unit/1e6:.2f} {det_slower*hertz_unit/1e6:.2f}", end = '                                                                            \r')
    eq = pylcp.rateeq(lasers(det_MOT, det_slower),permMagnetsPylcp, ham,include_mag_forces=False)
    try:
        eq.set_initial_pop(np.array([1., 0., 0., 0.]))
    except ValueError: # Quick and dirty solution to detect the two fermionic hamiltonians
        eq.set_initial_pop(np.array([0.5, 0.5, 0., 0., 0., 0., 0., 0.]))
    return findCaptureVelocity(np.array([-10,0,0]), eq)

def isCaptured(sol):
    # captured = -1
    # finalPosition = np.array([sol.r[i][-1] for i in range(3)])
    # finalVelocity = 0*np.array([sol.v[i][-1]*velocity_unit for i in range(3)])  # Fix capture cond
    # if (np.linalg.norm(finalPosition)**2 + np.linalg.norm(finalVelocity)**2 < 1e-2):
    #     #print('initial velocity: '+ str(sol.v[0][0]) +' captured')
    #     captured = 1 
    return weak_cap_cond(sol.v[:,-1],sol.r[:,-1])
    # return captured

def atomTrajectoryToMOT(v0, r0, eqn, angle = 0, classifier = isCaptured, tmax=10, max_step=m_step):
    eqn.set_initial_position_and_velocity(r0, np.array([v0*np.cos(angle),v0*np.sin(angle),0]))
    eqn.evolve_motion([0., 25e-3/time_unit], events=[captured_condition,lost_condition,backwards_lost],
                      max_step=max_step)
    return classifier(eqn.sol)

def findCaptureVelocity(r0,eqn):
    if(atomTrajectoryToMOT(2, r0, eqn, tmax=10, max_step=1)==-1):
        return 0
    return bisect(atomTrajectoryToMOT,2, 40.,
       args=(r0, eqn),
       xtol=1e-3, rtol=1e-3, full_output=False)


def captureVelocityForEq_ranged(det_MOT, det_slower, ham, *args, lasers = MOT_and_Slow_Beams, intervals = [0, 100/velocity_unit, 150/velocity_unit, 300/velocity_unit], angle = 0, rateeq_args = {}, **kwargs):
    print (f"{det_MOT*hertz_unit/1e6:.2f} {det_slower*hertz_unit/1e6:.2f}", end = '                                                                            \r')
    eq = pylcp.rateeq(lasers(det_MOT, det_slower,*args,**kwargs),permMagnetsPylcp, ham,include_mag_forces=False, **rateeq_args)
    try:
        eq.set_initial_pop(np.array([1., 0., 0., 0.]))
    except ValueError: # Quick and dirty solution to detect the two fermionic hamiltonians
        eq.set_initial_pop(np.array([0.5, 0.5, 0., 0., 0., 0., 0., 0.]))    
    # return findCaptureVelocityRange(np.array([-10,0,0]), eq, intervals, angle = angle)
    return findCaptureVelocityRange_fast(np.array([-10,0,0]), eq, angle = angle)

def findCaptureVelocityRange(r0, eqn, intervals = [0, 100/velocity_unit, 150/velocity_unit, 300/velocity_unit],angle = 0):
    signs = []
    roots = []
    signs.append(0 if atomTrajectoryToMOT(intervals[0], r0, eqn, tmax=10, max_step=m_step) < 1 else 1)
    for xlow, xhigh in zip(intervals[:-1], intervals[1:]):
        xhigh_sign = 0 if atomTrajectoryToMOT(xhigh, r0, eqn, angle = angle, tmax=25e-3/time_unit, max_step=m_step) < 1 else 1
        if (signs[-1] == xhigh_sign):
            continue

        roots.append(bisect(atomTrajectoryToMOT,xlow, xhigh,
                    args=(r0, eqn, angle),
                    xtol=1/velocity_unit, rtol=1e-3, full_output=False))
        signs.append(xhigh_sign)
    if (roots == []):
        return ([0],[0])
    return (roots, signs)

def capture_classifier(sol):
    if (sol.r[0,-1] > 1):
        return 1
    if (isCaptured(sol) > 0):
        return 0
    return -1

def findCaptureVelocityRange_fast(r0, eqn, angle = 0):
    vel_range = [-1/velocity_unit, 1000/velocity_unit]
    signs = []
    roots = []
    # signs.append(0 if atomTrajectoryToMOT(vel_range, r0, eqn, tmax=10, max_step=m_step) != 0 else 1)
    # for xlow, xhigh in zip(intervals[:-1], intervals[1:]):
    #     xhigh_sign = 0 if atomTrajectoryToMOT(xhigh, r0, eqn, angle = angle, tmax=25e-3/time_unit, max_step=m_step) < 1 else 1
    #     if (signs[-1] == xhigh_sign):
    #         continue

    #     roots.append(bisect(atomTrajectoryToMOT,xlow, xhigh,
    #                 args=(r0, eqn, angle),
    #                 xtol=1/velocity_unit, rtol=1e-3, full_output=False))
    #     signs.append(xhigh_sign)
        
    roots.append(bisect(atomTrajectoryToMOT, *vel_range, args=(r0, eqn, angle, lambda sol : capture_classifier(sol) + 0.5), xtol=1/velocity_unit, rtol=1e-3, full_output=False))
    # vel_range[0] = roots[0] - 10/velocity_unit
    roots.append(bisect(atomTrajectoryToMOT, *vel_range, args=(r0, eqn, angle, lambda sol : capture_classifier(sol) - 0.5), xtol=1/velocity_unit, rtol=1e-3, full_output=False))
    if (abs(roots[0] - roots[1]) < 2/velocity_unit):
        roots = []
    else:
        signs = [0,1,0]
    if (roots == []):
        return ([0],[0])
    return (roots, signs)


from scipy.stats import norm

mean = 150/velocity_unit
std = 30/velocity_unit

capture_cdf = lambda x : norm.cdf(x, mean, std)
capture_pdf = lambda x : norm.pdf(x, mean, std)

def convert_to_captured(roots, signs):
    global capture_cdf
    captured_percentage = 0
    for root_low, root_high, sign in zip(roots[:-1], roots[1:],signs[1:-1]):
        captured_percentage += sign*(capture_cdf (root_high) - capture_cdf (root_low))
    return captured_percentage

to_captured = lambda arr : np.array([convert_to_captured(*x) for x in arr])
to_captured_2D = lambda arr : np.array([[convert_to_captured(*x) for x in y] for y in arr]) # Write a more general solution
to_captured_ND = lambda arr : np.array([convert_to_captured(*x) if isinstance(x, tuple) else to_captured_ND(x) for x in arr])

    
def get_cap_range(roots, signs):
    for root_low, root_high, sign in zip(roots[:-1], roots[1:],signs[1:-1]):
        if sign:
            return [root_low, root_high]
    return [0,0]
    
to_cap_range = lambda arr : np.array([get_cap_range(*d) for d in arr])

if __name__ == "__main__":
    #Now if I make the radial and axial plots, they should still look as expected. Along the line y=-x, By =-Bx by symmetry.
    posns = np.arange(-20.0,20.0,0.02)
    fig, ax = plt.subplots(1,2,figsize=[10,5])
    ax[0].plot(posns,np.array([permMagnets.fieldCartesian(np.array([-i,0,0]),0)/(10**(-4)*consts.value('Bohr magneton')/(10**6*91*consts.h)) for i in posns]))
    ax[1].plot(posns,np.array([permMagnets.fieldCartesian(np.array([0.0,0.0,i]),0)/(10**(-4)*consts.value('Bohr magneton')/(10**6*91*consts.h)) for i in posns]))

    #checking that the pylcp still gets the same field profiles
    ax[0].plot(posns,np.array([permMagnetsPylcp.Field(np.array([-i,0,0]),0)/(10**(-4)*consts.value('Bohr magneton')/(10**6*91*consts.h)) for i in posns]), 'k--')
    ax[1].plot(posns,np.array([permMagnetsPylcp.Field(np.array([0.0,0.0,i]),0)/(10**(-4)*consts.value('Bohr magneton')/(10**6*91*consts.h)) for i in posns]), 'k--')