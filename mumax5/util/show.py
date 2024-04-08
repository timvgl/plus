"""Plotting helper functions."""

import matplotlib.pyplot as _plt
import numpy as _np

import mumax5 as _m5


def hsl_to_rgb(H, S, L):
    """Convert color from HSL to RGB."""
    Hp = _np.mod(H/(_np.pi/3.0), 6.0)
    C = _np.where(L<=0.5, 2*L*S, 2*(1-L)*S)
    X = C * (1 - _np.abs(_np.mod(Hp, 2.0) - 1.0))
    m = L - C / 2.0

    # R = m + X for 1<=Hp<2 or 4<=Hp<5
    # R = m + C for 0<=Hp<1 or 5<=Hp<6
    R = m + _np.select([((1<=Hp)&(Hp<2)) | ((4<=Hp)&(Hp<5)),
                        (Hp<1) | (5<=Hp)], [X, C], 0.)
    # G = m + X for 0<=Hp<1 or 3<=Hp<4
    # G = m + C for 1<=Hp<3
    G = m + _np.select([(Hp<1) | ((3<=Hp)&(Hp<4)),
                        (1<=Hp)&(Hp<3)], [X, C], 0.)
    # B = m + X for 2<=Hp<3 or 5<=Hp<6
    # B = m + C for 3<=Hp<5
    B = m + _np.select([((2<=Hp)&(Hp<3)) | (5<=Hp),
                        (3<=Hp)&(Hp<5)], [X, C], 0.)

    # clip rgb values to be in [0,1]
    R, G, B = _np.clip(R,0.,1.), _np.clip(G,0.,1.), _np.clip(B,0.,1.)

    return R, G, B


def vector_to_rgb(x, y, z):
    """Map vector (with norm<1) to RGB."""
    H = _np.arctan2(y, x)
    S = _np.sqrt(x ** 2 + y ** 2 + z ** 2)
    L = 0.5 + 0.5 * z
    return hsl_to_rgb(H, S, L)


def _quantity_img_xy_extent(quantity):
    """Return the extent for imshow images of xy cross sections..."""
    cx, cy, _ = quantity._impl.system.cellsize
    ox, oy, _ = quantity.grid.origin
    nx, ny, _ = quantity.grid.size
    extent = [
        cx * (ox - 0.5),
        cx * (ox - 0.5 + nx),
        cy * (oy - 0.5),
        cy * (oy - 0.5 + ny),
    ]
    return extent

def get_rgba(field, quantity, layer=0):
    # TODO also CUDAfy this function. This is exactly what GPUs are made for.
    field = field[:, layer]  # select the layer
    field /= _np.max(_np.linalg.norm(field, axis=0))  # rescale to make maximum norm 1

    # Create rgba image from the vector data
    _, ny, nx = field.shape  # image size
    rgba = _np.ones((ny, nx, 4))  # last index for R,G,B, and alpha channel
    rgba[:,:,0], rgba[:,:,1], rgba[:,:,2] = vector_to_rgb(field[0,:,:], field[1,:,:], field[2,:,:])

    # Set alpha channel to one inside the geometry, and zero outside
    rgba[:, :, 3] = quantity._impl.system.geometry[layer]
    return rgba


def show_field(quantity, layer=0):
    """Plot a mumax5.FieldQuantity with 3 or 6 components using the mumax3 colorscheme."""
    if not isinstance(quantity, _m5.FieldQuantity):
        raise TypeError("The first argument should be a FieldQuantity")
    
    if (quantity.ncomp != 3 and quantity.ncomp != 6):
        raise ValueError(
            "Can not create a vector field image because the field quantity "
            + "does not have 3 or 6 components."
        )

    field = quantity.eval()
    rgba = []
    if quantity.ncomp == 6:
        rgba = [get_rgba(field[0:3], quantity, layer), get_rgba(field[3:6], quantity, layer)]
        name = ['\n' + " (Sublattice 1)", '\n' + " (Sublattice 2)"]
    else:
        rgba = [get_rgba(field, quantity, layer)]
        name = [""]
    plotter(quantity, rgba, name)


def show_neel(quantity, layer=0):
    """Plot The Neel vector of an AFM using the mumax3 colorscheme."""
    if not isinstance(quantity, _m5.FieldQuantity):
        raise TypeError("The first argument should be a FieldQuantity")
    
    if (quantity.ncomp != 6):
        raise ValueError(
            "Can not create a Neel vector field image because the field quantity "
            + "does not have 6 components."
        )

    field = quantity.eval()
    neel_field = 0.5 * _np.subtract(field[0:3], field[3:6])

    rgba = [get_rgba(neel_field, quantity, layer)]

    plotter(quantity, rgba, [" (Neel vector field)"])

def plotter(quantity, rgba, name=[]):
    fig = _plt.figure()
    for i in range(len(rgba)):
        ax = fig.add_subplot(1, len(rgba), i+1)
        ax.set_title(quantity.name + name[i])
        ax.set_facecolor("gray")
        ax.imshow(rgba[i], origin="lower", extent=_quantity_img_xy_extent(quantity))
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
    _plt.show()



def show_layer(quantity, component=0, layer=0):
    """Visualize a single component of a mumax5.FieldQuantity."""
    if not isinstance(quantity, _m5.FieldQuantity):
        raise TypeError("The first argument should be a FieldQuantity")

    field = quantity.eval()
    field = field[component, layer]  # select component and layer

    geometry = quantity._impl.system.geometry[layer]
    field = _np.ma.array(field, mask=_np.invert(geometry))

    cmap = _plt.get_cmap("viridis")
    cmap.set_bad(alpha=0.0)  # This will affect cells outside the mask (i.e. geometry)

    fig = _plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("gray")
    ax.set_title(quantity.name + ", component=%d" % component)
    ax.imshow(
        field, cmap=cmap, origin="lower", extent=_quantity_img_xy_extent(quantity)
    )
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    _plt.show()

def show_neel_quiver(quantity, title=''):
    
    # Still needs some fudging...

    field = quantity.eval()
    m1, m2 = field[0:3], field[3:6]
    mx1, my1, mz1 = m1[0][0], m1[1][0], m1[2][0]
    mx2, my2, mz2 = m2[0][0], m2[1][0], m2[2][0]

    mx = mx1 + mx2
    my = my1 + my2
    mz = mz1 + mz2

    # Normalizing to [-1, +1]
    if mx.max() > 1 or mx.min() < 1:
        mxx = 2 * (mx - mx.min()) / (mx.max() - mx.min()) - 1
    if my.max() > 1 or my.min() < 1:
        myy = 2 * (my - my.min()) / (my.max() - my.min()) - 1
    if mz.max() > 1 or mz.min() < 1:
        mzz = 2 * (mz - mz.min()) / (mz.max() - mz.min()) - 1
    
    #Plotting
    cmap = _plt.get_cmap('jet')
    norm = _plt.Normalize(mzz.min(), mzz.max())
    fig, ax = _plt.subplots()
    ax.quiver(mxx, myy)

    cax = ax.imshow(mzz, cmap=cmap, interpolation='none')
    cbar = _plt.colorbar(_plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)

    cbar.set_label('z-comp')
    if title:
        ax.set_title(title)
    _plt.show()
