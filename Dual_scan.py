from init import * # Run everything in init.py

MOT_range = np.arange(-750e6/hertz_unit, 2100e6/hertz_unit + 10e6/hertz_unit,10e6/hertz_unit)
slower_range = np.arange(-1700e6/hertz_unit, 1700e6/hertz_unit + 50e6/hertz_unit,50e6/hertz_unit)
# MOT_range = np.arange(-100e6/hertz_unit, 200e6/hertz_unit + 10e6/hertz_unit,10e6/hertz_unit)
# slower_range = np.arange(-1000e6/hertz_unit, 500e6/hertz_unit + 100e6/hertz_unit,100e6/hertz_unit)

import multiprocessing as mp

capture_data = {}
c_data = []

def run_cap_vel_scan(i):
    # print(f"\n{i}:")
    cap_vels = []
    for dSlow in slower_range:
        print (f"{dSlow*hertz_unit/1e6:.2f}\r",end="")
        cap_vels.append([captureVelocityForEq_ranged(dMOT, dSlow, Hamiltonians[i]) for dMOT in MOT_range])
    return (i,cap_vels)

if __name__ == "__main__":
    with mp.Pool(processes=8) as pool:
        c_data =  pool.map(run_cap_vel_scan, Hamiltonians.keys())
    # c_data =  map(run_cap_vel_scan, Hamiltonians.keys())

    for i, cdata in c_data:
        capture_data[i] = cdata

    np.savez('data/capture_data_new_cap_alg.npz',capture_data=c_data,MOT_range=MOT_range,slower_range=slower_range)

    to_be_plotted = None

    for i, cap_data in capture_data.items():
        if to_be_plotted is None:
            to_be_plotted = abundance_data[i]*to_captured_2D(cap_data)
            continue
        to_be_plotted += abundance_data[i]*to_captured_2D(cap_data)
        
    plt.figure(figsize=[20,20])
    plt.axes().set_aspect('equal')
    plt.pcolormesh(*np.meshgrid(MOT_range*hertz_unit/1e6, slower_range*hertz_unit/1e6), to_be_plotted, cmap = 'gnuplot')
    plt.xlabel("MOT detuning -$\\nu_{114}$ [MHz]")
    plt.ylabel("Slower detuning -$\\nu_{114}$ [MHz]")
    cbar = plt.colorbar()
    cbar.set_label("MOT signal [a.u.]")
    plt.show()

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    color_norm = Normalize(0,1)
    fig, axs = plt.subplots(2,len(capture_data.values())//2,figsize=[60,30],sharex=True, sharey=True)
    fig.subplots_adjust(right=0.9)
    axs = axs.T.flatten()
    for i, c_data in enumerate(capture_data.items()):

        axs[i].set_aspect('equal')
        axs[i].set_title(c_data[0])
        colormesh = axs[i].pcolormesh(*np.meshgrid(MOT_range*hertz_unit/1e6, slower_range*hertz_unit/1e6), to_captured_2D(c_data[1]), cmap = 'gnuplot', norm = color_norm)
        if i in [1,3,5,7]:
            axs[i].set_xlabel("MOT detuning -$\\nu_{114}$ [MHz]")
        if i in [0,1]:
            axs[i].set_ylabel("Slower detuning -$\\nu_{114}$ [MHz]")
    cbar_ax = fig.add_axes([0.9, 0.15, 0.0125, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm = color_norm, cmap = 'gnuplot'), cax = cbar_ax)
    cbar.set_label("MOT signal [a.u.]")

    plt.show()

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    color_norm = Normalize(0,1)
    fig, axs = plt.subplots(2,len(capture_data.values())//2,figsize=[30,15],sharex=True, sharey=True)
    fig.subplots_adjust(right=0.9)
    axs = axs.T.flatten()
    for i, c_data in enumerate(capture_data.items()):

        axs[i].set_aspect('equal')
        axs[i].set_title(c_data[0])
        colormesh = axs[i].pcolormesh(*np.meshgrid(MOT_range*hertz_unit/1e6, slower_range*hertz_unit/1e6), abundance_data[c_data[0]]*to_captured_2D(c_data[1]), cmap = 'gnuplot', norm = color_norm)
        if i in [1,3,5,7]:
            axs[i].set_xlabel("MOT detuning -$\\nu_{114}$ [MHz]")
        if i in [0,1]:
            axs[i].set_ylabel("Slower detuning -$\\nu_{114}$ [MHz]")
    cbar_ax = fig.add_axes([0.9, 0.15, 0.0125, 0.7])
    cbar = fig.colorbar(ScalarMappable(norm = color_norm, cmap = 'gnuplot'), cax = cbar_ax)
    cbar.set_label("MOT signal [a.u.]")

    plt.show()

    # from matplotlib.colors import Normalize
    # from matplotlib.cm import ScalarMappable

    # color_norm = Normalize(0,250)
    # fig, axs = plt.subplots(2,len(capture_data.values())//2,figsize=[60,30],sharex=True, sharey=True)
    # fig.subplots_adjust(right=0.9)
    # axs = axs.T.flatten()
    # for i, c_data in enumerate(capture_data.items()):

    #     axs[i].set_aspect('equal')
    #     axs[i].set_title(c_data[0])
    #     colormesh = axs[i].pcolormesh(*np.meshgrid(MOT_range*hertz_unit/1e6, slower_range*hertz_unit/1e6), np.array(c_data[1])*velocity_unit, cmap = 'gnuplot', norm = color_norm)
    #     if i in [1,3,5,7]:
    #         axs[i].set_xlabel("MOT detuning -$\\nu_{114}$ [MHz]")
    #     if i in [0,1]:
    #         axs[i].set_ylabel("Slower detuning -$\\nu_{114}$ [MHz]")
    # cbar_ax = fig.add_axes([0.9, 0.15, 0.0125, 0.7])
    # cbar = fig.colorbar(ScalarMappable(norm = color_norm, cmap = 'gnuplot'), cax = cbar_ax)
    # cbar.set_label("Capture velocity [m/s]")

    # plt.show()