from init import * # Run everything in init.py

MOT_range = np.linspace(-1700e6/hertz_unit, 1700e6/hertz_unit,401)
slower_range = np.linspace(-1700e6/hertz_unit, 1700e6/hertz_unit,401)

slower_capture_data = {}

beams = [MOT_and_Slow_Beams, MOT_and_Slow_Beams_sig_2, MOT_and_Slow_Beams_lin]
relevant_isotopes = [114, 116, 113]

for i in beams:
    print("\n1")
    slower_capture_data[i] = {}
    for h in relevant_isotopes:
        print(f"\n{h}:")
        slower_capture_data[i][h] = [captureVelocityForEq_ranged(MOT_detuning, dSlower, Hamiltonians[h],lasers=i) for dSlower in slower_range]

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