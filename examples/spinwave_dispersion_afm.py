import matplotlib.pyplot as plt
import numpy as np

from mumax5 import Ferromagnet, Grid, World

# Antiferromagnetic spinwave dispersion relation

# Numerical parameters
fmax = 5E13          # maximum frequency (in Hz) of the sinc pulse
T = 2E-12            # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)  # the sample time
dx = 1E-9            # cellsize
nx = 512             # number of cells

# Material/system parameters
Bz = 0.2           # bias field along the z direction
A = 10E-12         # exchange constant
A_nn = -5E-12
A_c = -400E-12
Ms = 400e3         # saturation magnetization
alpha = 0.005      # damping parameter
gamma = 1.76E11    # gyromagnetic ratio
K = 1e3
mu0 = 1.256637062E-6

# Create the world
grid_size = (nx, 1, 1)
cell_size = (dx, dx, dx)

world = World(cell_size)
world.bias_magnetic_field = (0, 0, Bz)

# Create a ferromagnet
magnet = Ferromagnet(world, Grid(size=grid_size), 6)
magnet.msat = Ms
magnet.msat2 = Ms
magnet.aex = A
magnet.aex2 = A
magnet.afmex_nn = A_nn
magnet.afmex_cell = A_c
magnet.alpha = alpha
magnet.latcon = dx

magnet.ku1 = K
magnet.ku12 = K
magnet.anisU = (0, 0, 1)


Bt = lambda t: (1e2* np.sinc(2 * fmax * (t - T / 2)), 0, 0)
mask = np.zeros(shape=(1, 1, nx))
# Put signal at the center of the simulation box
mask[:, :, nx // 2 - 1:nx // 2 + 1] = 1
magnet.bias_magnetic_field.add_time_term(Bt, mask)

magnet.magnetization = (0, 0, 1, 0, 0, -1)
magnet.minimize()

timepoints = np.linspace(0, T, 1 + int(T / dt))
outputquantities = {'m': lambda: magnet.magnetization.eval()}

# Run solver
output = world.timesolver.solve(timepoints, outputquantities)

# Apply the two dimensional FFT
m = np.array(output['m'], dtype=float)
mx = m[:, 0, 0, 0, :]
mx_fft = np.fft.fft2(mx)
mx_fft = np.fft.fftshift(mx_fft)

plt.figure(figsize=(10, 6))

# Show the intensity plot of the 2D FFT
extent = [-(2 * np.pi) / (2 * dx), (2 * np.pi) / (2 * dx), -1 /
          (2 * dt), 1 / (2 * dt)]  # extent of k values and frequencies

plt.imshow(np.abs(mx_fft)**2, extent=extent,
           aspect='auto', origin='lower', cmap="inferno")

# Plot the analytical derived dispersion relation
k = np.linspace(-np.pi/dx, np.pi/dx, 250)
mu0Ms = mu0 * Ms
He = -4 * A_c / (dx*dx*mu0Ms)
Ha = 2 * K / mu0Ms
Hint = (2*A - A_nn) / mu0Ms
freq_theory = mu0 * gamma / (2*np.pi) * np.sqrt((2*He + Ha + Hint * k**2) * (Ha + Hint * k**2)) + Bz * gamma/(2*np.pi)
plt.plot(k, freq_theory, '--', lw=1)

plt.axhline(gamma * Bz / (2 * np.pi), c='g', ls='--', lw=1)
plt.xlim([-np.pi/dx, np.pi/dx])
plt.ylim([0, fmax])
plt.vlines([-1/dx, 1/dx], 0, fmax, 'r', '--', lw=1)
plt.ylabel("$f$ (Hz)")
plt.xlabel("$k$ (1/m)")
plt.colorbar()
plt.title('Antiferromagnetic Spinwave Dispersion Relation')
plt.show()
