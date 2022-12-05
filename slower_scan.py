from init import * # Run everything in init.py

MOT_range = np.linspace(-1700e6/hertz_unit, 1700e6/hertz_unit,51)
slower_range = np.linspace(-1700e6/hertz_unit, 1700e6/hertz_unit,51)

slower_capture_data = {}
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

beams = [MOT_and_Slow_Beams_timed2, MOT_and_Slow_Beams_sig_2_timed, MOT_and_Slow_Beams_lin_timed]
relevant_isotopes = [114, 116, 113]

def run_beam(beam):
    ret = {}
    for h in relevant_isotopes:
        print(f"\n{h}:")
        ret[h] = [captureVelocityForEq_ranged(MOT_detuning, dSlower, Hamiltonians[h],lasers=beam) for dSlower in slower_range]
    return beam, ret

if __name__ == "__main__":
    import multiprocessing as mp
    with mp.Pool(processes = 8) as pool:
        cdata = pool.map(run_beam,beams)
    
    slower_capture_data = {}
    
    for x,y in cdata:
        slower_capture_data[x] = y


    fig, ax = plt.subplots(1,1,figsize=[10,6])
    cols = ['b','g','r']
    labels = ["$\\sigma_-$","$\\sigma_+$","linear"]
    for i, beam_cap_data in enumerate(slower_capture_data.values()):
        plt.plot(slower_range*hertz_unit/1e6, np.sum([to_captured(x)*abundance_data[key] for key, x in beam_cap_data.items()],axis=0),label=labels[i], c = cols[i])
    ax.grid()
    ax.set_xlabel("Slower detuning [MHz]")
    ax.set_ylabel("MOT Signal [a.u.]")
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    plt.show()