"""Classes for common shapes and their manipulation."""

import numpy as _np
from scipy.spatial import Delaunay as _Delaunay

# ==================================================
# Parent Shape class

class Shape:
    """Base class for all shapes using constructive solid geometry (CSG).
    This mutable class holds and manipulates a given shape function.

    Parameters
    ----------
    shape_func : Callable[[x,y,z], bool]
        Function returning True if (x,y,z) is within this shape.
    """
    def __init__(self, shape_func=(lambda x,y,z: False)):
        self.shape_func = shape_func

    def __call__(self, x, y, z):
        """Returns True if (x,y,z) is within this shape.
        Calling shape.shape_func(x,y,z) or shape(x,y,z) is the same."""
        return self.shape_func(x, y, z)

    # -------------------------
    # transformations on this shape

    def transform4(self, transform_matrix):
        """Transform this shape according to a given 4x4 matrix,
        which can represent any affine transformation (rotate, scale, shear,
        translate). It is usually of the form [[R, T], [0, 1]], with R a
        3x3 rotation matrix and T a 3x1 translation vector.
        Returns transformed self.
        """
        old_func = self.shape_func  # copy old version of self
        def new_func(x,y,z):
            coord_vec = _np.array([x, y, z, _np.ones_like(x)])
            x_,y_,z_,_ = _np.tensordot(transform_matrix, coord_vec, axes=1)
            return old_func(x_,y_,z_)
        self.shape_func = new_func
        return self

    def transform3(self, transform_matrix):
        """Transform this shape according to a given 3x3 matrix (rotate, scale,
        shear).
        Returns transformed self.
        """
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x,y,z: old_func(*_np.tensordot(transform_matrix,
                                                  _np.array([x, y, z]), axes=1))
        return self
        

    def transform(self, transform_matrix):
        """Transform this shape according to a given 3x3 matrix (rotate, scale,
        shear) or 4x4 matrix (like 3x3 plus translations). It is usually of the
        form [[R, T], [0, 1]], with R a 3x3 rotation matrix and T a 3x1
        translation vector.
        Returns transformed self.
        """
        if transform_matrix.shape == (4, 4):
            return self.transform4(transform_matrix)
        return self.transform3(transform_matrix)

    def rotate_x(self, theta):
        """Rotate this shape theta radians counter-clockwise around the x-axis."""
        rotmat = _np.array([[1, 0, 0],
                            [0, _np.cos(theta), _np.sin(theta)],
                            [0, -_np.sin(theta), _np.cos(theta)]])
        return self.transform3(rotmat)

    def rotate_y(self, theta):
        """Rotate this shape theta radians counter-clockwise around the y-axis."""
        rotmat = _np.array([[_np.cos(theta), 0, -_np.sin(theta)],
                            [0, 1, 0],
                            [_np.sin(theta), 0, _np.cos(theta)]])
        return self.transform3(rotmat)

    def rotate_z(self, theta):
        """Rotate this shape theta radians counter-clockwise around the z-axis."""
        rotmat = _np.array([[_np.cos(theta), _np.sin(theta), 0],
                            [-_np.sin(theta), _np.cos(theta), 0],
                            [0, 0, 1]])
        return self.transform3(rotmat)

    def translate(self, dx, dy, dz):
        """Translate this shape by the vector (dx,dy,dz)."""
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x,y,z: old_func(x-dx, y-dy, z-dz)
        return self

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
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x,y,z: old_func(x/sx, y/sy, z/sz)
        return self

    # -------------------------
    # operations on this shape

    def invert(self):
        """Invert this shape (logical NOT)."""
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x,y,z: _np.logical_not(old_func(x, y, z))
        return self

    def repeat(self, min_point, max_point):
        """Infinitely repeat everything from this shape enclosed in a bounding
        box defined by the given minimum and maximum points, while everything
        outside this box is ignored.

        Parameters
        ----------
        min_point : tuple[float] of size 3
            Smallest x, y and z coordinates of the bounding box (inclusive).
        max_point : tuple[float] of size 3
            Largest x, y and z coordinates of the bounding box (exclusive).
        
        Setting any coordinate to None will not repeat the shape in this direction.
        """
        def none_mod(x, x_min, x_max):
            if x_min is None or x_max is None:
                return x
            return (x-x_min)%(x_max-x_min) + x_min

        x0, y0, z0 = min_point
        x1, y1, z1 = max_point
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x,y,z: old_func(none_mod(x, x0, x1),
                                                 none_mod(y, y0, y1),
                                                 none_mod(z, z0, z1))
        return self

    # -------------------------
    # operations on single shape returning new shape

    def __neg__(self):
        """Returns a new shape as the inverse of given shape (logical NOT)."""
        return Shape(lambda x,y,z: _np.logical_not(self(x, y, z)))

    def copy(self):
        """Returns a new shape which is a copy of this shape."""
        func_copy = self.shape_func
        return Shape(func_copy)
        
    # -------------------------
    # operations between shapes altering this shape
    
    def add(self, other: "Shape"):
        """Add given shape to this shape (logical OR).
        Calling a.add(b), a+=b or a|=b is the same."""
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x, y, z: old_func(x, y, z) | other(x, y, z)
        return self

    def __iadd__(self, other: "Shape"):
        """Add given shape to this shape (logical OR).
        Calling a.add(b), a+=b or a|=b is the same."""
        return self.add(other)

    def __ior__(self, other: "Shape"):
        """Add given shape to this shape (logical OR).
        Calling a.add(b), a+=b or a|=b is the same."""
        return self.add(other)

    def sub(self, other: "Shape"):
        """Subtract given shape from this shape (logical AND NOT).
        Calling a.sub(b) or a-=b is the same."""
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x,y,z: old_func(x,y,z) & _np.logical_not(other(x,y,z))
        return self
    
    def __isub__(self, other: "Shape"):
        """Subtract given shape from this shape (logical AND NOT).
        Calling a.sub(b) or a-=b is the same."""
        return self.sub(other)

    def intersect(self, other: "Shape"):
        """Intersect given shape with this shape (logical AND).
        Calling a.intersect(b), a&=b and a/=b are the same."""
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x,y,z: old_func(x,y,z) & other(x,y,z)
        return self
    
    def __iand__(self, other: "Shape"):
        """Intersect given shape with this shape (logical AND).
        Calling a.intersect(b), a&=b and a/=b are the same."""
        return self.intersect(other)

    def __itruediv__(self, other: "Shape"):
        """Intersect given shape with this shape (logical AND).
        Calling a.intersect(b), a&=b and a/=b are the same."""
        return self.intersect(other)

    def xor(self, other: "Shape"):
        """Keep everything from this shape and the given shape, except the
        intersection (logical XOR).
        Calling a.xor(b) or a^=b is the same."""
        old_func = self.shape_func  # copy old version of self
        self.shape_func = lambda x, y, z: old_func(x, y, z) ^ other(x, y, z)
        return self

    def __ixor__(self, other: "Shape"):
        """Keep everything from this shape and the given shape, except the
        intersection (logical XOR).
        Calling a.xor(b) or a^=b is the same."""
        return self.xor(other)
    
    # -------------------------
    # operations between shapes returning new shape

    def __add__(self, other: "Shape"):
        """Returns new shape as union of given shapes (logical OR).
        Calling a+b or a|b is the same."""
        return Shape(lambda x, y, z: self(x, y, z) | other(x, y, z))

    def __or__(self, other: "Shape"):
        """Returns new shape as union of given shapes (logical OR).
        Calling a+b or a|b is the same."""
        return a+b

    def __sub__(self, other: "Shape"):
        """Returns new shape as the first shape with the second shape removed
        (logical AND NOT)."""
        return Shape(lambda x,y,z: self(x,y,z) & _np.logical_not(other(x,y,z)))

    def __and__(self, other: "Shape"):
        """Returns new shape as intersection of given shapes (logical AND).
        Calling a&b or a/b is the same."""
        return Shape(lambda x, y, z: self(x, y, z) & other(x, y, z))

    def __truediv__(self, other: "Shape"):
        """Returns new shape as intersection of given shapes (logical AND).
        Calling a&b or a/b is the same."""
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
    """Sphere with given diameter or radius."""
    def __init__(self, diam=None, radius=None):
        if radius is not None:
            diam = radius
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

class Cone(Shape):
    """3D cone with the vertex down. It has a given diameter at a given height."""
    def __init__(self, diam, height):
        def shape_func(x, y, z):
            return (z >= 0) & ((x/diam)**2 + (y/diam)**2 <= 0.25*(z/height)**2)
        super().__init__(shape_func)

class Cylinder(Shape):
    """Cylinder along z with given diameter and height."""
    def __init__(self, diam, height):
        def shape_func(x, y, z):
            return (z <= 0.5*height) & (z >= -0.5*height) & \
                   ((x/diam)**2+(y/diam)**2<=0.25)
        super().__init__(shape_func)

class Cuboid(Shape):
    """3D rectangular slab with given sides, including minimum, excluding maximum."""
    def __init__(self, sidex, sidey, sidez):
        def shape_func(x, y, z):
            rx, ry, rz = 0.5*sidex, 0.5*sidey, 0.5*sidez
            return (-rx <= x)&(x < rx) & (-ry <= y)&(y < ry) & (-rz <= z)&(z < rz)
        super().__init__(shape_func)

class Cube(Cuboid):
    """Cube with given side length."""
    def __init__(self, side):
        super().__init__(side, side, side)

class Rectangle(Shape):
    """2D Rectangle in the xy-plane with given sides."""
    def __init__(self, sidex, sidey):
        def shape_func(x, y, z):
            rx, ry = 0.5*sidex, 0.5*sidey
            return (-rx <= x)&(x < rx) & (-ry <= y)&(y < ry)
        super().__init__(shape_func)

class Square(Rectangle):
    """Square with given side length."""
    def __init__(self, side):
        super().__init__(side, side)


class XRange(Shape):
    """Range of x-values: xmin <= x < xmax"""
    def __init__(self, xmin, xmax):
        super().__init__(lambda x,y,z: (xmin <= x) & (x < xmax))

class YRange(Shape):
    """Range of y-values: ymin <= y < ymax"""
    def __init__(self, ymin, ymax):
        super().__init__(lambda x,y,z: (ymin <= y) & (y < ymax))

class ZRange(Shape):
    """Range of z-values: zmin <= z < zmax"""
    def __init__(self, zmin, zmax):
        super().__init__(lambda x,y,z: (zmin <= z) & (z < zmax))

class Torus(Shape):
    """Torus with given major and minor diameters.
    
    Parameters
    ----------
    major_diam : float
        Distance between opposite centers of the tube.
    minor_diam : float
        Diameter of the tube.

    The torus is major_diam + minor_diam wide and minor_diam high.
    When major_diam = minor_diam, there will be no hole.
    """
    def __init__(self, major_diam, minor_diam):
        D, d = major_diam, minor_diam
        def shape_func(x, y, z):
            return (x**2 + y**2 + z**2 + 0.25*D**2 - 0.25*d**2)**2 <= D**2*(x**2 + y**2)
        super().__init__(shape_func)

# =========================
# Convex polyhedra

class DelaunayHull(Shape):
    """The Delaunay hull of a list of 3D points. These points can serve as the
    vertices of a convex polyhedron.
    
    Parameters
    ----------
    points : ndarray of double, shape (npoints, 3)
    """
    def __init__(self, points):
        self.hull = _Delaunay(points)
        def shape_func(x, y, z):
            return self.hull.find_simplex(_np.stack([x,y,z], axis=-1)) >= 0
        super().__init__(shape_func)

class Tetrahedron(DelaunayHull):
    """Tetrahedron (4-faced platonic solid) where all vertices lie on a sphere
    with the given diameter."""
    def __init__(self, diam):
        d_circumsphere = 2
        vertices = diam*_np.asarray([[2*_np.sqrt(2)/3, 0, -1/3],
                                    [-_np.sqrt(2)/3, _np.sqrt(2/3), -1/3],
                                    [-_np.sqrt(2)/3, -_np.sqrt(2/3), -1/3],
                                    [0, 0, 1]])
        super().__init__(vertices/d_circumsphere)

class Octahedron(Shape):  # Shape, not DelauneyHull, as there is a simple formula
    """Octahedron (8-faced platonic solid) where all vertices lie on a sphere
    with the given diameter."""
    def __init__(self, diam):
        super().__init__(lambda x,y,z: _np.abs(x)+_np.abs(y)+_np.abs(z) <= 0.5*diam)

class Dodecahedron(DelaunayHull):
    """Dodecahedron (12-faced platonic solid) where all vertices lie on a sphere
    with the given diameter."""
    def __init__(self, diam):
        phi = (1 + _np.sqrt(5))/2
        d_circumsphere = 2*_np.sqrt(3)
        vertices = diam*_np.asarray([[1, 1, 1], [1, 1, -1], [1, -1, 1],
                                     [1, -1, -1], [-1, 1, 1], [-1, 1, -1],
                                     [-1, -1, 1], [-1, -1, -1], [0, phi, 1/phi],
                                     [0, phi, -1/phi], [0, -phi, 1/phi],
                                     [0, -phi, -1/phi], [1/phi, 0, phi],
                                     [1/phi, 0, -phi], [-1/phi, 0, phi],
                                     [-1/phi, 0, -phi], [phi, 1/phi, 0],
                                     [phi, -1/phi, 0], [-phi, 1/phi, 0],
                                     [-phi, -1/phi, 0]])
        super().__init__(vertices/d_circumsphere)

class Icosahedron(DelaunayHull):
    """Dodecahedron (20-faced platonic solid) where all vertices lie on a sphere
    with the given diameter."""
    def __init__(self, diam):
        phi = (1 + _np.sqrt(5))/2
        d_circumsphere = 2*_np.sqrt(phi**2 + 1)
        vertices = diam*_np.asarray([[0, 1, phi], [0, 1, -phi], [0, -1, phi],
                                     [0, -1, -phi], [phi, 0, 1], [phi, 0, -1],
                                     [-phi, 0, 1], [-phi, 0, -1], [1, phi, 0],
                                     [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0]])
        super().__init__(vertices/d_circumsphere)

class Icosidodecahedron(DelaunayHull):
    """Icosidodecahedron  where all vertices lie on a sphere with the given
    diameter."""
    def __init__(self, diam):
        phi = (1 + _np.sqrt(5))/2
        phiSq = phi*phi
        a = 1
        d_circumsphere = 4/(_np.sqrt(5) - 1)
        vertices = [[0, 0, phi], [0, 0, -phi]] + [[i*.5, j*phi/2, k*phiSq/2] \
                    for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
        vertices += [[y,z,x] for x,y,z in vertices] + [[z, x, y] for x,y,z in vertices]
        super().__init__(diam*_np.asarray(vertices)/d_circumsphere)


# ==================================================
# TODO List of Mumax3 shapes to add
# Layers
# Layer
# Cell
# ImageShape
# GrainRoughness

# TODO Bonus shapes
# Polygons
# Regular polygons

# ==================================================

if __name__=="__main__":

    import pyvista as pv

    shape = Tetrahedron(1)
    shape.translate_z(0.5)

    res = 101
    a = 1
    x = y = z = _np.linspace(-a, a, res)

    def plot_shape_3D(shape, x, y, z, title=""):
        """Show a shape given x, y and z coordinate arrays. This uses PyVista."""
        X, Y, Z = _np.meshgrid(x, y, z, indexing="ij")  # the logical indexing
        S = shape(X, Y, Z)
        dx, dy, dz = (x[1]-x[0]), (y[1]-y[0]), (z[1]-z[0])
                     # [::-1] for [x,y,z] not [z,y,x] and +1 for cells, not points
        image_data = pv.ImageData(dimensions=(len(x)+1, len(y)+1, len(z)+1),  
                     spacing=(dx,dy,dz), origin=(x[0]-0.5*dx, y[0]-0.5*dy, z[0]-0.5*dz))
        image_data.cell_data["values"] = _np.float32(S.flatten("F"))
        threshed = image_data.threshold_percent(0.5)  # only show True

        plotter = pv.Plotter()
        plotter.add_mesh(threshed, color="white", show_edges=True,
                         show_scalar_bar=False, smooth_shading=True)
        plotter.show_axes()
        plotter.show_bounds()
        if len(title) > 0: plotter.add_title(title)
        plotter.show()

    plot_shape_3D(shape, x,y,z)

