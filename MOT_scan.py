from init import * # Run everything in init.py

MOT_range = np.linspace(-750e6/hertz_unit, 2100e6/hertz_unit,401)

slower_det = -175e6/hertz_unit

time_cutoff = 0.35e-3/time_unit

mean = 180/velocity_unit
std = 10/velocity_unit

capture_cdf = lambda x : norm.cdf(x, mean, std)

def convert_to_captured(roots, signs):
    global capture_cdf
    captured_percentage = 0
    for root_low, root_high, sign in zip(roots[:-1], roots[1:],signs[1:-1]):
        captured_percentage += sign*(capture_cdf (root_high) - capture_cdf (root_low))
    return captured_percentage

to_captured = lambda arr : np.array([convert_to_captured(*x) for x in arr])
to_captured_2D = lambda arr : np.array([[convert_to_captured(*x) for x in y] for y in arr]) # Write a more general solution

def run_MOT_scan(i):
    print (i, "S")
    return i, [captureVelocityForEq_ranged(dMOT, slower_det, Hamiltonians[i], time_cutoff, lasers = MOT_and_Slow_Beams_timed) for dMOT in MOT_range]
def run_MOT_scan_wo_slower(i):
    print (i, "W")
    return i, [captureVelocityForEq_ranged(dMOT, slower_det, Hamiltonians[i], lasers = MOT_Beams) for dMOT in MOT_range]

if __name__ == "__main__":
    __spec__ = None
    import multiprocessing as mp

    with mp.Pool(processes=8) as pool:
        cdata = pool.map_async(run_MOT_scan, isotope_shifts.keys())
        cdata_wo_slower = pool.map(run_MOT_scan_wo_slower, isotope_shifts.keys())
        cdata = cdata.get()
    
    MOT_capture_data = {}

    for x, y in cdata:
        MOT_capture_data[x] = y

    # with mp.Pool(processes=8) as pool:
        
    MOT_capture_data_wo_slower = {}

    for x, y in cdata_wo_slower:
        MOT_capture_data_wo_slower[x] = y

    # for i in isotope_shifts.keys():
    #     print (f"\n{i}:")
    #     MOT_capture_data[i] =  [captureVelocityForEq_ranged(dMOT, slower_det, Hamiltonians[i]) for dMOT in MOT_range]

    def get_cap_range(roots, signs):
        for root_low, root_high, sign in zip(roots[:-1], roots[1:],signs[1:-1]):
            if sign:
                return [root_low, root_high]
        return [0,0]
        
    to_cap_range = lambda arr : np.array([get_cap_range(*d) for d in arr])

    fig, axs =  plt.subplots(2,len(MOT_capture_data.keys())//2,figsize = [25,10])
    axs = axs.flatten()
    for i, c_data in enumerate(MOT_capture_data.items()):
        # axs[i].plot(MOT_range*hertz_unit/1e6, to_captured(c_data[1]))
        axs[i].fill_between(MOT_range*hertz_unit/1e6, to_cap_range(c_data[1])[:,0]*velocity_unit, to_cap_range(c_data[1])[:,1]*velocity_unit)
        axs[i].set_title(c_data[0])
        axs[i].xaxis.set_minor_locator(MultipleLocator(100))
    plt.show()

    fig, axs =  plt.subplots(2,len(MOT_capture_data.keys())//2,figsize = [25,10])
    axs = axs.flatten()
    for i, c_data in enumerate(MOT_capture_data.items()):
        axs[i].plot(MOT_range*hertz_unit/1e6, to_captured(c_data[1]))
        axs[i].set_title(c_data[0])
        axs[i].xaxis.set_minor_locator(MultipleLocator(100))

    plt.show()

    to_be_plotted = None

    fig, ax = plt.subplots(1,1, figsize = [10,6])

    for i, cap_data in MOT_capture_data.items():
        if to_be_plotted is None:
            to_be_plotted = abundance_data[i]*to_captured(cap_data)
            continue
        to_be_plotted += abundance_data[i]*to_captured(cap_data)

    ax.plot(MOT_range*hertz_unit/1e6, to_be_plotted)
    ax.grid()
    ax.xaxis.set_minor_locator(MultipleLocator(100))

    plt.show()

    # MOT_capture_data_wo_slower = {}

    # for i in isotope_shifts.keys():
    #     print (f"\n{i}:")
    #     MOT_capture_data_wo_slower[i] =  [captureVelocityForEq(dMOT, slower_det, Hamiltonians[i], lasers = MOT_Beams) for dMOT in MOT_range]

    # fig, axs =  plt.subplots(2,len(MOT_capture_data.keys())//2,figsize = [25,10])
    # axs = axs.flatten()
    # for i, c_data in enumerate(MOT_capture_data_wo_slower.items()):
    #     axs[i].plot(MOT_range*hertz_unit/1e6, np.array(c_data[1])*velocity_unit)
    #     axs[i].set_title(c_data[0])

    # plt.show()

    to_be_plotted = None

    for i, cap_data in MOT_capture_data.items():
        if to_be_plotted is None:
            to_be_plotted = abundance_data[i]*to_captured(cap_data)
            continue
        to_be_plotted += abundance_data[i]*to_captured(cap_data)
        
    fig, ax = plt.subplots(1,1,figsize=[20,12])
    ax.plot(MOT_range*hertz_unit/1e6, to_be_plotted, label = "With slower")

    to_be_plotted = None

    for i, cap_data in MOT_capture_data_wo_slower.items():
        if to_be_plotted is None:
            to_be_plotted = abundance_data[i]*to_captured(cap_data)
            continue
        to_be_plotted += abundance_data[i]*to_captured(cap_data)

    ax.plot(MOT_range*hertz_unit/1e6, to_be_plotted, label = "Without slower")
    ax.set_xlabel("MOT detuning - $\\nu_{114}$ [MHz]")
    ax.set_ylabel("MOT signal [a.u.]")
    ax.legend()
    ax.grid()
    ax.xaxis.set_minor_locator(MultipleLocator(100))

    plt.show()

    w_slower = None
    wo_slower = None

    for (i, cap_data), (j, cap_data_wo_slower) in zip(MOT_capture_data.items(), MOT_capture_data_wo_slower.items()):
        if w_slower is None:
            w_slower = abundance_data[i]*to_captured(cap_data)
            wo_slower = abundance_data[i]*to_captured(cap_data)
            continue
        w_slower += abundance_data[i]*to_captured(cap_data)
        wo_slower += abundance_data[i]*to_captured(cap_data)

    epsilon = 1e-4
        
    to_be_plotted = w_slower + epsilon
    to_be_plotted/=wo_slower + epsilon
    # to_be_plotted[wo_slower<=epsilon] =0

    fig, ax = plt.subplots(1,1,figsize=[20,12])
    ax.plot(MOT_range*hertz_unit/1e6, to_be_plotted)
    ax.set_xlabel("MOT detuning - $\\nu_{114}$ [MHz]")
    ax.set_ylabel("Ratio")
    ax.grid()
    ax.xaxis.set_minor_locator(MultipleLocator(100))

    plt.show()