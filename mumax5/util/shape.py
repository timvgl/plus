"""Classes for common shapes and their manipulation."""

import numpy as _np

# ==================================================
# Parent Shape class

class Shape:
    """Base class for all shapes.

    Parameters
    ----------
    shape_func : Callable[[x,y,z], bool]
        Function returning True if (x,y,z) is within the base shape,
        without any transformation.
    transform_matrix : 4x4 ndarray (default=identity(4))
        4x4 matrix representing the transformation of this shape,
        usually of the form [[R, T], [0, 1]], with R a 3x3 rotation matrix
        and T a 3x1 translation vector.
    """
    def __init__(self, shape_func=(lambda x,y,z: False),
                 transform_matrix=_np.identity(4)):
        self.shape_func = shape_func
        self.transform_matrix = transform_matrix

    def at(self, x, y, z):
        """Returns True if (x,y,z) is within this transformed shape.
        Calling shape.at(x,y,z) or shape(x,y,z) is the same."""
        coord_vec = _np.array([x, y, z, _np.ones_like(x)])
        x_,y_,z_,_ = _np.tensordot(self.transform_matrix, coord_vec, axes=1)
        return self.shape_func(x_, y_, z_)

    def __call__(self, x, y, z):
        """Returns True if (x,y,z) is within this transformed shape.
        Calling shape.at(x,y,z) or shape(x,y,z) is the same."""
        return self.at(x, y, z)
        
    # -------------------------
    # transformations on itself

    def transform(self, mat):
        """Transform this shape according to a given 4x4 matrix,
        usually of the form [[R, T],[0, 1]], with R a 3x3 rotation matrix
        and T a 3x1 translation vector.
        Returns transformed self."""
        self.transform_matrix = self.transform_matrix @ mat
        return self

    def rotate_matrix(self, rotmat):
        """Rotate this shape according to the given 3x3 rotation matrix."""
        if rotmat.shape == (3,3):
            rotmat = _np.pad(rotmat, (0,1))  # 3x3 to 4x4        
            rotmat[3,3] = 1
        return self.transform(rotmat)

    def rotate_x(self, theta):
        """Rotate this shape theta radians counter-clockwise around the x-axis."""
        rotmat = _np.array([[1, 0, 0],
                            [0, _np.cos(theta), _np.sin(theta)],
                            [0, -_np.sin(theta), _np.cos(theta)]])
        return self.rotate_matrix(rotmat)

    def rotate_y(self, theta):
        """Rotate this shape theta radians counter-clockwise around the y-axis."""
        rotmat = _np.array([[_np.cos(theta), 0, -_np.sin(theta)],
                            [0, 1, 0],
                            [_np.sin(theta), 0, _np.cos(theta)]])
        return self.rotate_matrix(rotmat)

    def rotate_z(self, theta):
        """Rotate this shape theta radians counter-clockwise around the z-axis."""
        rotmat = _np.array([[_np.cos(theta), _np.sin(theta), 0],
                            [-_np.sin(theta), _np.cos(theta), 0],
                            [0, 0, 1]])
        return self.rotate_matrix(rotmat)

    def translate(self, dx, dy, dz):
        """Translate this shape by the vector (dx,dy,dz)."""
        mat = _np.identity(4)
        mat[0:3,3] = (-dx,-dy,-dz)
        return self.transform(mat)

    def translate_x(self, dx):
        """Translate this shape by dx along the x-axis."""
        return self.translate(dx, 0, 0)
    
    def translate_y(self, dy):
        """Translate this shape by dy along the y-axis."""
        return self.translate(0, dy, 0)
    
    def translate_z(self, dz):
        """Translate this shape by dz along the z-axis."""
        return self.translate(0, 0, dz)

    def scale(self, sx, sy=None, sz=1):
        """Scale this shape, using (0,0,0) as the origin.
        Takes 1, 2 or 3 arguments:
            1. (s): scale by s in all directions.
            2. (sx, sy): scale by sx and sy in the xy-plane, but do not scale z.
            3. (sx, sy, sz): scale by sx, sy and sz in the x-, y- and
            z-direction respectively.
        """
        if sy is None:
            sy = sz = sx
        mat = _np.diag((1/sx, 1/sy, 1/sz, 1))
        return self.transform(mat)

    # -------------------------
    # operations on single shape returning shape
    # TODO I don't like that these return a new shape instance,
    # but others mutate self

    def get_inverse(self):
        """Returns a new shape as the inverse of this shape (logical NOT).
        shape.get_inverse() and -shape is the same."""
        return Shape(lambda x, y, z: _np.logical_not(self(x, y, z)))

    def __neg__(self):
        """Returns a new shape as the inverse of this shape (logical NOT).
        shape.get_inverse() and -shape is the same."""
        return self.get_inverse()

    def get_repeat(self, px, py, pz):
        """Returns a new Shape which repeats everything of this shape between
        points (0,0,0) to (px,py,pz) infinitely, while everything outside this
        box is ignored.

        Parameters
        ----------
        px, py, pz : floats
        Period of repitition in each direction.
        Setting p_i to None will not repeat the shape in this direction.
        """
        nm = lambda x, p: x if p is None else x%p  # nm for None Modulo

        return Shape(lambda x,y,z: self(nm(x,px), nm(y,py), nm(z,pz)))
        

    def __mod__(self, periods):
        """Calling shape % p is the same as shape.get_repeat(p,p,p).
        Calling shape % (px,py,pz) is the same as shape.get_repeat(px,py,pz).
        """
        if hasattr(periods, "__iter__"):
            px, py, pz = periods
        else:  # single number
            px = py = pz = periods

        return self.get_repeat(px, py, pz)
        
    
    # -------------------------
    # operations between shapes

    def __add__(self, other: "Shape"):
        """Returns new shape as union of given shapes (logical OR)."""
        return Shape(lambda x, y, z: self(x, y, z) | other(x, y, z))

    def __sub__(self, other: "Shape"):
        """Returns new shape as the first shape with the second shape removed
        (logical AND NOT)."""
        return Shape(lambda x,y,z: self(x,y,z) & _np.logical_not(other(x,y,z)))

    def __and__(self, other: "Shape"):
        """Returns new shape as intersection of given shapes (logical AND).
        a&b and a/b are the same."""
        return Shape(lambda x, y, z: self(x, y, z) & other(x, y, z))

    def __truediv__(self, other: "Shape"):
        """Returns new shape as intersection of given shapes (logical AND).
        a&b and a/b are the same."""
        return self & other

    def __xor__(self, other: "Shape"):
        """Returns a new shape which is everything from both shapes, except
        their intersection (logical XOR)."""
        return Shape(lambda x, y, z: self(x, y, z) ^ other(x, y, z))


# ==================================================
# Child shapes

class Empty(Shape):
    """Empty space."""
    def __init__(self):
        super().__init__(lambda x,y,z: False)

class Universe(Shape):
    """All of space."""
    def __init__(self):
        super().__init__(lambda x,y,z: True)

class Ellipsoid(Shape):
    """Ellipsoid with given diameters diamx, diamy, diamz."""
    def __init__(self, diamx, diamy, diamz):
        def shape_func(x, y, z):
            return (x/diamx)**2 + (y/diamy)**2 + (z/diamz)**2 <= 0.25
        super().__init__(shape_func)

class Sphere(Ellipsoid):
    """Sphere with given diameter."""
    def __init__(self, diam):
        super().__init__(diam, diam, diam)

class Ellipse(Shape):
    """Ellipse in the xy-plane with given diameters diamx and diamy."""
    def __init__(self, diamx, diamy):
        def shape_func(x, y, z):
            return (x/diamx)**2 + (y/diamy)**2 <= 0.25
        super().__init__(shape_func)

class Circle(Ellipse):
    """Circle in the xy-plane with given diameter."""
    def __init__(self, diam):
        super().__init__(diam, diam)


# ==================================================
# TODO List of Mumax3 shapes to add

# Cone
# Cylinder
# Circle
# Cuboid
# Rect
# Square
# XRange
# YRange
# ZRange
# Layers
# Layer
# Cell
# ImageShape
# GrainRoughness

# ==================================================

if __name__=="__main__":

    import matplotlib.pyplot as plt

    shape = Ellipsoid(1, 0.75, 0.5).translate(0.5, 0.75/2, 0.25)
    shape = shape.get_repeat(1, 1, 1)


    res = 50
    a = 2
    x = _np.linspace(-a, a, res)
    y = _np.linspace(-a, a, res)
    z = _np.linspace(-a, a, res)
    X, Y, Z = _np.meshgrid(x, y, z, indexing="ij")

    geom = shape(X, Y, Z)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(geom)
    ax.axis("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    plt.show()
