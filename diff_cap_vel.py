from init import *
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

theta_range = np.linspace(1e-6,10e-3+1e-6,20)
phi_range = np.linspace(0,2*np.pi,72, endpoint=False)

CORES = 16

# def MOT_Beams_modified(det_MOT, *args):
#     return pylcp.laserBeams([
#         {'kvec':np.array([-np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
#         {'kvec':np.array([-np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
#         {'kvec':np.array([np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
#         {'kvec':np.array([np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy*2,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
#     ], beam_type=pylcp.gaussianBeam)

# def MOT_Beams_modified(det_MOT, *args):
#     return pylcp.laserBeams([
#         {'kvec':np.array([-1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([1/np.sqrt(2), -1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
#     ], beam_type=pylcp.gaussianBeam)

# def MOT_Beams_modified(det_MOT, *args):
#     return pylcp.laserBeams([
#         {'kvec':np.array([-np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([-np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([-1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([-1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
#     ], beam_type=pylcp.gaussianBeam)

# def MOT_Beams_modified(det_MOT, *args):
#     return pylcp.laserBeams([
#         {'kvec':np.array([1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([-1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([-1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
#         {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
#     ], beam_type=pylcp.gaussianBeam)
    
def MOT_Beams_modified(det_MOT, *args):
    return pylcp.laserBeams([
        {'kvec':np.array([1/2, np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-np.sqrt(3)/2, 1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([-1/2, -np.sqrt(3)/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([np.sqrt(3)/2, -1/2, 0.]), 'pol':-1, 'delta':det_MOT, 's':2*MOT_s_xy,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0.,  1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width},
        {'kvec':np.array([0., 0., -1.]), 'pol':+1, 'delta':det_MOT, 's':MOT_s_z,'wb':MOT_beam_width}
    ], beam_type=pylcp.gaussianBeam)


def run(args):
    global progress
    with progress.get_lock():
        progress.value += 1
        if progress.value % 100 == 0:
            print(f"{progress.value}/{total_runtime}: {100*progress.value/total_runtime:.2f}%")
    H = Hamiltonians[112]
    det_MOT = -0.5
    
    eq = pylcp.rateeq(MOT_Beams_modified(det_MOT + isotope_shifts[112]), permMagnetsPylcp, H, include_mag_forces=False)
    eq.set_initial_pop([1,0,0,0])
    return findCaptureVelocityRange_fast(np.array([-45.5,0,0]),eq, angle=args)


def init_worker(pgr, t_r):
    global progress, total_runtime
    progress = pgr
    total_runtime = t_r

def plot(data):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    cmesh = ax.pcolormesh(*np.meshgrid(phi_range, theta_range, indexing='ij'),data, cmap = 'gnuplot',shading="nearest")
    fig.colorbar(cmesh)
    plt.show()
#     color_norm = Normalize(0,1)
#     fig, axs = plt.subplots(2,4)
#     axs = axs.T.flatten()
#     for i, ax, d in zip(Hamiltonians.keys(), axs, data):
#         ax.axes.set_aspect('equal')
#         ax.pcolormesh(*(np.array(np.meshgrid(slower_range*hertz_unit/1e6, MOT_range*hertz_unit/1e6))),abundance_data[i]*d, cmap = 'gnuplot', norm = color_norm,shading='nearest')
#         ax.set_title(i)
#     cbar_ax = fig.add_axes([0.9, 0.15, 0.0125, 0.7])
#     cbar = fig.colorbar(ScalarMappable(norm = color_norm, cmap = 'gnuplot'), cax = cbar_ax)
#     plt.show()
#     fig2, (ax2, axc) = plt.subplots(1,2)
#     ax2.axes.set_aspect('equal')
#     tbp = np.sum([abundance_data[i]*d for i,d in zip(Hamiltonians.keys(),data)],axis = 0)
#     cmesh = ax2.pcolormesh(*np.meshgrid(slower_range*hertz_unit/1e6, MOT_range*hertz_unit/1e6),tbp.T, cmap = 'gnuplot')
#     ax2.set_xlabel("MOT detuning -$\\nu_{112}$ [MHz]")
#     ax2.set_ylabel("Slower detuning -$\\nu_{112}$ [MHz]")
#     fig2.colorbar(cmesh,ax=axc)
#     plt.show()

if __name__ == "__main__":

    params = np.array(np.meshgrid(theta_range, phi_range, indexing='ij')).T.reshape([-1,2])
    params_reg_unit = params
    __spec__ = None
    import multiprocessing as mp
    total_runtime = len(theta_range)*len(phi_range)
    progress = mp.Value('i', 0)

    with mp.Pool(processes=16, initializer=init_worker, initargs=(progress,total_runtime)) as pool:
        data = np.array(pool.map(run, params))
  
    saveable_data = np.array([d[0][-1]*velocity_unit for d in data])
    proc_data = saveable_data.reshape([len(phi_range),len(theta_range)])
    data = data.reshape([len(phi_range),len(theta_range),2])
    np.savez("out.npz",phis=phi_range, thetas=theta_range, data=proc_data)
    plot(proc_data)