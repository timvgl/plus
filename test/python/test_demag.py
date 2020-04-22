from mumax5.engine import World, Grid

import numpy as np


def demag_field_py(magnet):
    kernel = magnet._demagkernel()
    mag = magnet.msat * magnet.magnetization.get()
    # add padding to the magnetization so that the size of magnetization
    # matches the size of the kernel
    pad = ((0, kernel.shape[1]-mag.shape[1]),
           (0, kernel.shape[2]-mag.shape[2]),
           (0, kernel.shape[3]-mag.shape[3]))
    m = np.pad(mag, ((0, 0), *pad), 'constant')

    # fourier transform of the magnetization and the kernel
    m = np.fft.fftn(m, axes=(1, 2, 3))
    kxx, kyy, kzz, kxy, kxz, kyz = np.fft.fftn(kernel, axes=(1, 2, 3))

    # apply the kernel and perform inverse fft
    hx = np.fft.ifftn(m[0]*kxx+m[1]*kxy+m[2]*kxz)
    hy = np.fft.ifftn(m[0]*kxy+m[1]*kyy+m[2]*kyz)
    hz = np.fft.ifftn(m[0]*kxz+m[1]*kyz+m[2]*kzz)

    # return the real part
    mu0 = 4*np.pi*1e-7
    h = -mu0*np.array([hx, hy, hz]).real
    return h[:,
             (h.shape[1]-mag.shape[1]):,
             (h.shape[2]-mag.shape[2]):,
             (h.shape[3]-mag.shape[3]):]


class TestDemag:
    def test_demagfield(self):
        world = World((1e-9, 1e-9, 1e-9))
        magnet = world.add_ferromagnet(Grid((16, 4, 3)))
        wanted = demag_field_py(magnet)
        result = magnet.demag_field.eval()
        err = np.max(np.abs((wanted-result)/result))
        assert err < 1e-3
