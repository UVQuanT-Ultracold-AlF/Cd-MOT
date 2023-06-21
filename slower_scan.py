from init import * # Run everything in init.py
from sys import argv

MOT_range = np.linspace(-1700e6/hertz_unit, 1700e6/hertz_unit,51)
slower_range = np.linspace(-1400e6/hertz_unit, 600e6/hertz_unit,51)

time_range = 0.5e-3/time_unit

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

beams = [MOT_and_Slow_Beams, MOT_and_Slow_Beams_sig_2, MOT_and_Slow_Beams_lin]
relevant_isotopes = [112]#,111,113]#, 116, 113]
powers = [0.4,0.15,0.1,0.05]

def run_beam(beam):
    ret = {}
    h = relevant_isotopes[0]
    # p = powers[0]
    # for h in relevant_isotopes:
    for p in powers:
        mod_slower_s(p)
        print(f"\n{h} {p}:")
        # ret[h]
        ret[p] = [captureVelocityForEq_ranged(-175e6/hertz_unit + isotope_shifts[112], dSlower + isotope_shifts[112], Hamiltonians[h],lasers=beam, intervals=np.linspace(1/velocity_unit,500/velocity_unit,50), angle=0) for dSlower in slower_range]
    return beam, ret
def plot_slower(slower_cap_data, mean = 150/velocity_unit, std = 30/velocity_unit, *args, c_cdf = None):
    capture_cdf = lambda x : norm.cdf(x, mean, std) if c_cdf is None else c_cdf

    def convert_to_captured(roots, signs):
        nonlocal capture_cdf
        captured_percentage = 0
        for root_low, root_high, sign in zip(roots[:-1], roots[1:],signs[1:-1]):
            captured_percentage += sign*(capture_cdf (root_high) - capture_cdf (root_low))
        return captured_percentage

    to_captured = lambda arr : np.array([convert_to_captured(*x) for x in arr])
    to_captured_2D = lambda arr : np.array([[convert_to_captured(*x) for x in y] for y in arr]) # Write a more general solution
        
    fig, axs =  plt.subplots(len(beams),len(powers)*len(relevant_isotopes),figsize = [25,10], sharex= True, sharey= True)
    #axs = axs.flatten()
    for i, c_data in enumerate(slower_cap_data.items()):
        for j, cdata in enumerate(c_data[1].items()):
            # axs[i].plot(MOT_range*hertz_unit/1e6, to_captured(c_data[1]))
            if (len(slower_cap_data)) == 1:
                ax = axs[j]
            elif len(c_data[1]) == 1:
                ax = axs[i]
            else:
                ax = axs[i][j]
            ax.fill_between(slower_range*hertz_unit/1e6, to_cap_range(cdata[1])[:,0]*velocity_unit, to_cap_range(cdata[1])[:,1]*velocity_unit, step='mid')
            ax.set_title(f"{cdata[0]} {c_data[0]}")
            ax.xaxis.set_minor_locator(MultipleLocator(100))
    plt.xlabel("Slower Detuning [MHz]")
    plt.show()


    fig, ax = plt.subplots(1,1,figsize=[10,6])
    cols = ['b','g','r']
    labels = ["$\\sigma_-$","$\\sigma_+$","linear"]
    for i, beam_cap_data in enumerate(slower_cap_data.values()):
        plt.plot(slower_range*hertz_unit/1e6, np.sum([to_captured(x)*1 for key, x in beam_cap_data.items()],axis=0),label=labels[i], c = cols[i])
        # plt.plot(slower_range*hertz_unit/1e6, np.sum([to_captured(x)*abundance_data[key] for key, x in beam_cap_data.items()],axis=0),label=labels[i], c = cols[i])
    ax.grid()
    ax.set_xlabel("Slower detuning [MHz]")
    ax.set_ylabel("MOT Signal [a.u.]")
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    plt.show()

if __name__ == "__main__":
    if len(argv) <= 1:
        __spec__ = None # This is needed due to an ipython bug on windows
        import multiprocessing as mp
        with mp.Pool(processes = 8) as pool:
            cdata = pool.map(run_beam,beams)
        
        slower_capture_data = {}
        
        for x,y in cdata:
            slower_capture_data[x] = y

    plot_slower(slower_capture_data)