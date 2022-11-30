from init import * # Run everything in init.py

MOT_range = np.arange(-550e6/hertz_unit, 50e6/hertz_unit + 10e6/hertz_unit,10e6/hertz_unit)
slower_range = np.arange(-1200e6/hertz_unit, 500e6/hertz_unit + 50e6/hertz_unit,50e6/hertz_unit)
# MOT_range = np.arange(-100e6/hertz_unit, 200e6/hertz_unit + 10e6/hertz_unit,10e6/hertz_unit)
# slower_range = np.arange(-1000e6/hertz_unit, 500e6/hertz_unit + 100e6/hertz_unit,100e6/hertz_unit)
angle_range = np.linspace(0.,np.pi/16, 8)

import multiprocessing as mp

capture_data = {}
c_data = []

def run_cap_vel_scan(i):
    # print(f"\n{i}:")
    cap_vels = []
    for dSlow in slower_range:
        cap_vels.append([captureVelocityForEq_ranged(dMOT, dSlow, Hamiltonians[114], angle = i) for dMOT in MOT_range])
    return (i,cap_vels)

if __name__ == "__main__":
    with mp.Pool(processes=8) as pool:
        c_data =  pool.map(run_cap_vel_scan, angle_range)
    # c_data =  map(run_cap_vel_scan, Hamiltonians.keys())

    for i, cdata in c_data:
        capture_data[i] = cdata

    to_be_plotted = None

    for i, cap_data in capture_data.items():
        if to_be_plotted is None:
            to_be_plotted = to_captured_2D(cap_data)/len(angle_range)
            continue
        to_be_plotted += to_captured_2D(cap_data)/len(angle_range)
        
    plt.figure(figsize=[20,20])
    plt.axes().set_aspect('equal')
    plt.pcolormesh(*np.meshgrid(MOT_range*hertz_unit/1e6, slower_range*hertz_unit/1e6), to_be_plotted, cmap = 'gnuplot')
    plt.xlabel("MOT detuning -$\\nu_{114}$ [MHz]")
    plt.ylabel("Slower detuning -$\\nu_{114}$ [MHz]")
    cbar = plt.colorbar()
    cbar.set_label("MOT signal [a.u.]")
    plt.show()
