from init import *

slower_magnet = permMagnetsPylcp

dr = 0.02
dv = 2/velocity_unit

r = np.arange(-10, 10+dr, 2*dr)
v = np.arange(-100/velocity_unit, 250/velocity_unit, 2*dv)

R, V = np.meshgrid(r, v)

eq = pylcp.rateeq(MOT_and_Slow_Beams(-175e6/hertz_unit,-700e6/hertz_unit),slower_magnet, Hamiltonians[114],include_mag_forces=False,)
eq.generate_force_profile([R, np.zeros(R.shape), np.zeros(R.shape)],
                           [V, np.zeros(V.shape), np.zeros(V.shape)],
                           name='Frad', progress_bar=True)

fig, ax = plt.subplots()
im = ax.imshow(-1*eq.profile['Frad'].F[0]*(1+0*hbar*k*91e6/1.89301e-25/1e3), origin='lower',
           extent=(np.amin(r)/100, np.amax(r)/100,
                   np.amin(v*velocity_unit)-dv/2, np.amax(v*velocity_unit)-dv/2),
           aspect='auto', cmap='viridis')
ax.set_xlabel("Position [m]")
ax.set_ylabel("Velocity [m/s]")
cb1 = plt.colorbar(im)
cb1.set_label("Force [$\hbar k\Gamma$]")
plt.tight_layout()
plt.show()
