from init import * # Run everything in init.py

MOT_range = np.arange(isotope_shifts[113]-550e6/hertz_unit, isotope_shifts[113]+50e6/hertz_unit,float(25e6/hertz_unit))
slower_range = np.arange(isotope_shifts[113]-1200e6/hertz_unit, isotope_shifts[113]+500e6/hertz_unit,float(100e6/hertz_unit))
# MOT_range = np.arange(-100e6/hertz_unit, 200e6/hertz_unit + 10e6/hertz_unit,10e6/hertz_unit)
# slower_range = np.arange(-1000e6/hertz_unit, 500e6/hertz_unit + 100e6/hertz_unit,100e6/hertz_unit)
angle_range = np.linspace(0.,np.pi/16, 8)
# angle_range = [0]
# time_range = np.linspace(0e-4/time_unit,1e-3/time_unit,8)
time_range = [0.5e-3/time_unit]

import multiprocessing as mp

capture_data = {}
c_data = []

def run_cap_vel_scan(i):
    # print(f"\n{i}:")
    cap_vels = []
    for t in time_range:
        cap_vels.append([[captureVelocityForEq_ranged(dMOT, dSlow, Hamiltonians[113], t, angle = i, lasers = MOT_and_Slow_Beams_timed) for dMOT in MOT_range] for dSlow in slower_range])
    return (i,cap_vels)

if __name__ == "__main__":
    with mp.Pool(processes=8) as pool:
        c_data =  pool.map(run_cap_vel_scan, angle_range)
    # c_data =  map(run_cap_vel_scan, Hamiltonians.keys())

    for i, cdata in c_data:
        capture_data[i] = cdata

    to_be_plotted = None

    for i, cap_data in capture_data.items():
        for d in  cap_data:
            if to_be_plotted is None:
                to_be_plotted = to_captured_2D(d)/len(angle_range)/len(time_range)
                continue
            to_be_plotted += to_captured_2D(d)/len(angle_range)/len(time_range)
        
    plt.figure(figsize=[5,10])
    # plt.axes().set_aspect('equal')
    plt.pcolormesh(*np.meshgrid(MOT_range*hertz_unit/1e6, slower_range*hertz_unit/1e6), to_be_plotted, cmap = 'gnuplot')
    plt.xlabel("MOT detuning -$\\nu_{114}$ [MHz]")
    plt.ylabel("Slower detuning -$\\nu_{114}$ [MHz]")
    cbar = plt.colorbar()
    cbar.set_label("MOT signal [a.u.]")
    plt.show()
