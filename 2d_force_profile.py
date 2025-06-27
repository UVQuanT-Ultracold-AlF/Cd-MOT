from init import *
from sys import argv

MOT_det = -0.5

def MOT_Beams_modified1(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
        {'kvec':np.array([-np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
        {'kvec':np.array([np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
        {'kvec':np.array([np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_Beams_modified2(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_Beams_modified3(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_Beams_modified4(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
    ], beam_type=pylcp.gaussianBeam)

def MOT_Beams_modified5(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':0.1,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':0.1,'wb':MOT_beam_width},
        {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':0.1,'wb':MOT_beam_width},
        {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':0.1,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':0.05,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':0.05,'wb':MOT_beam_width}
    ], beam_type=pylcp.gaussianBeam)

rateeq = pylcp.rateeq(MOT_Beams_modified2(MOT_det), permMagnetsPylcp, Hamiltonians[114] ,include_mag_forces=False)

rx = np.linspace(-2/cm_unit, 2/cm_unit, 200)
ry = np.linspace(-2/cm_unit, 2/cm_unit, 200)

RX, RY = np.meshgrid(rx, ry)
rateeq.generate_force_profile([RX, RY, np.zeros(RX.shape)],
                           [0,0,0],
                           name='Frad', progress_bar=True)

angles = np.linspace(-20e-3,20e-3,40)
vel = 50/velocity_unit

sols = []

for angle in angles:
    print(f"Angle: {angle*1000} mrad")
    rateeq.set_initial_position_and_velocity(np.array([-45/cm_unit, 0., 0.]),np.array([vel*np.cos(angle), vel*np.sin(angle), 0.]))
    rateeq.set_initial_pop(np.array([1., 0., 0., 0.]))
    rateeq.evolve_motion([0., 5e-2/time_unit], events=[captured_condition, lost_condition, backwards_lost], progress_bar=True, max_step = 1)
    sols.append(rateeq.sol)

fig, ax = plt.subplots(1, 1)
for sol in sols:
    ax.plot(sol.r[0],sol.r[1],'g-' if isCaptured(sol) == 1 else "r-")
colormesh = ax.pcolormesh(RX, RY,  np.einsum("i...,i...->...",np.array([rateeq.profile['Frad'].F[0],rateeq.profile['Frad'].F[1]]),np.array([rateeq.profile['Frad'].R[0],rateeq.profile['Frad'].R[1]]))/np.linalg.norm(np.array([rateeq.profile['Frad'].R[0],rateeq.profile['Frad'].R[1]]), axis=0), cmap = 'viridis', vmin=-0.15, vmax=0.15)
cb1 = plt.colorbar(colormesh)
cb1.set_label('$f(\hbar k \Gamma)$')
ax.set_xlabel('$x$ (cm)')
ax.set_ylabel('$y$ (cm)')
fig.subplots_adjust(left=0.12,right=0.9)
ax.set_aspect(1)
ax.set_xlim([-2,2])
ax.set_ylim([-2,2])