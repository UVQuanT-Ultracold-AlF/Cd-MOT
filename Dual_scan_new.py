from init import *
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# MOT_range = np.arange(-750e6/hertz_unit - isotope_shifts[112], 2100e6/hertz_unit - isotope_shifts[112] + 10e6/hertz_unit,10e6/hertz_unit)
# slower_range = np.arange(-1700e6/hertz_unit - isotope_shifts[112], 1700e6/hertz_unit - isotope_shifts[112] + 50e6/hertz_unit,50e6/hertz_unit)
MOT_range = np.arange(-1200e6/hertz_unit, 1600e6/hertz_unit + 50e6/hertz_unit,50e6/hertz_unit)
slower_range = np.arange(-2500e6/hertz_unit, 1500e6/hertz_unit + 50e6/hertz_unit,50e6/hertz_unit)
CORES = 16

def run(args):
    global progress
    with progress.get_lock():
        progress.value += 1
        if progress.value % 100 == 0:
            print(f"{progress.value}/{total_runtime}: {100*progress.value/total_runtime:.2f}%")
    H = Hamiltonians[args[2]]
    det_MOT = args[0]
    det_slower = args[1]
    
    eq = pylcp.rateeq(MOT_and_Slow_Beams_timed2(det_MOT + isotope_shifts[112], det_slower + isotope_shifts[112]),permMagnetsPylcp,H, include_mag_forces=False)
    if args[2] in [111,113]:
        eq.set_initial_pop([0.5,0.5,0,0,0,0,0,0])
    else:
        eq.set_initial_pop([1,0,0,0])
    if isotope_shifts[args[2]] + 0e6/hertz_unit < det_MOT + isotope_shifts[112] or isotope_shifts[args[2]] - 400e6/hertz_unit > det_MOT + isotope_shifts[112]:
        return ([0],[0])
    return findCaptureVelocityRange_fast(np.array([-45.5,0,0]),eq)


def init_worker(pgr, t_r):
    global progress, total_runtime
    progress = pgr
    total_runtime = t_r

def plot(data):
    color_norm = Normalize(0,1)
    fig, axs = plt.subplots(2,4)
    axs = axs.T.flatten()
    for i, ax, d in zip(Hamiltonians.keys(), axs, data):
        ax.axes.set_aspect('equal')
        ax.pcolormesh(*(np.array(np.meshgrid(slower_range*hertz_unit/1e6, MOT_range*hertz_unit/1e6))),abundance_data[i]*d, cmap = 'gnuplot', norm = color_norm,shading='nearest')
        ax.set_title(i)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.0125, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm = color_norm, cmap = 'gnuplot'), cax = cbar_ax)
    plt.show()
    fig2, (ax2, axc) = plt.subplots(1,2)
    ax2.axes.set_aspect('equal')
    tbp = np.sum([abundance_data[i]*d for i,d in zip(Hamiltonians.keys(),data)],axis = 0)
    cmesh = ax2.pcolormesh(*np.meshgrid(slower_range*hertz_unit/1e6, MOT_range*hertz_unit/1e6),tbp.T, cmap = 'gnuplot')
    ax2.set_xlabel("MOT detuning -$\\nu_{112}$ [MHz]")
    ax2.set_ylabel("Slower detuning -$\\nu_{112}$ [MHz]")
    fig2.colorbar(cmesh,ax=axc)
    plt.show()

if __name__ == "__main__":

    params = np.array(np.meshgrid(MOT_range, slower_range, list(Hamiltonians.keys()),indexing='ij')).T.reshape([-1,3])
    params_reg_unit = np.array(np.meshgrid(MOT_range*hertz_unit/1e6, slower_range*hertz_unit/1e6, list(Hamiltonians.keys()),indexing='ij')).T.reshape([-1,3])
    __spec__ = None
    import multiprocessing as mp
    total_runtime = len(list(Hamiltonians.keys()))*len(MOT_range)*len(slower_range)
    progress = mp.Value('i', 0)

    with mp.Pool(processes=16, initializer=init_worker, initargs=(progress,total_runtime)) as pool:
        data = np.array(pool.map(run, params))
  
    capture_data = to_captured(data)
    saveable_data = np.array([[d[0][0]*velocity_unit, d[0][1]*velocity_unit] if len(d[1]) != 1 else [-1,-1] for d in data])
    np.savetxt('capture_data.csv',np.column_stack((*(params_reg_unit.T),capture_data)),fmt = ('%5.5f','%5.5f','%d','%5.5f'),header=f"Capture data. mu = {mean*velocity_unit} m/s, std = {std*velocity_unit} m/s\n Frequency relative to 112\nSlower detuning [MHz], MOT detuning [MHz], Isotope, Captured proporiton [a.u]")
    np.savetxt('capture_data_raw.csv',np.column_stack((*(params_reg_unit.T),*(saveable_data.T))),fmt = ('%5.5f','%5.5f','%d','%5.5f','%5.5f'),header=f"Capture range data\n Frequency relative to 112\nSlower detuning [MHz], MOT detuning [MHz], Isotope, Lower capture velocity [m/s], Upper capture velocity [m/s]")
    capture_data = capture_data.reshape([len(list(Hamiltonians.keys())),len(slower_range),len(MOT_range)])
    data = data.reshape([len(list(Hamiltonians.keys())),len(slower_range),len(MOT_range),2])
    plot(capture_data)