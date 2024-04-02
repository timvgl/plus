import matplotlib.pyplot as plt
import numpy as np

from mumax5 import Ferromagnet, Grid, World
from mumax5.util import show_field

# Antiferromagnetic spinwave dispersion relation

# Numerical parameters
fmax = 6E13          # maximum frequency (in Hz) of the sinc pulse
T = 2E-12             # simulation time (longer -> better frequency resolution)
dt = 1 / (2 * fmax)  # the sample time
dx = 1E-9           # cellsize
nx = 512            # number of cells

# Material/system parameters
Bz = 0.2           # bias field along the z direction
A = 10E-12         # exchange constant
A_nn = -5E-12
A_c = -100E-12
Ms = 400e3         # saturation magnetization
alpha = 0.001      # damping parameter
gamma = 1.76E11    # gyromagnetic ratio
K = 1e3
a = 0.35e-9

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
k = np.linspace(-3e9, 3e9, 10000)
freq_theory = (2/(a*a)) * (2*A - A_nn) * gamma * np.abs(np.sin(k*dx/2)) / (np.pi * Ms) + gamma * Bz / (2 * np.pi)
plt.plot(k, freq_theory, 'g--', lw=1)

plt.xlim([-3e9, 3e9])
plt.ylim([0, fmax])
plt.ylabel("$f$ (Hz)")
plt.xlabel("$k$ (1/m)")
plt.colorbar()
plt.title('Antiferromagnetic Spinwave Dispersion Relation')
plt.show()
