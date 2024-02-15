import numpy as np
import sympy as sym
import sympy.geometry.point as symPoint
import sympy.geometry.line as symLine
import sympy.geometry.curve as symCurve

# TODO: Remove the dependency of __entity__ in the future
from .__entity__ import GeometryEntity
from sympy.utilities.iterables import is_sequence
from sympy.core.containers import Tuple

import polytex.mesh as ms
import polytex as ptx
import pyvista as pv

import os


class Point(np.ndarray):
    """
    A point / vector class inheriting from numpy.ndarray. This class is
    used to represent points in n-dimensional space. The shape of the
    array is (m, n), where m is the number of points and n is the
    dimensionality of the space.

    Notes
    ----------
    Efficient Vector / Point class in Python from:
    https://stackoverflow.com/questions/19458291/efficient-vector-point-class-in-python

    Examples
    --------
    >>> from polytex.geometry import Point
    >>> import numpy as np
    >>> p1 = Point([1, 2, 3])
    >>> p2 = Point([4, 5, 6])
    >>> p3 = Point([[1, 2, 3], [1, 2, 3]])
    >>> p4 = Point([[2, 4, 6], [2, 4, 6]])
    >>> dist34 = np.linalg.norm(p3 - p4, axis=1)
    >>> x, y, z = p1.x, p1.y, p1.z
    >>> x, y, z
    (1, 2, 3)
    >>> xyz = p1.xyz
    >>> assert (p1.xyz == np.array([1, 2, 3])).all()
    >>> # Operations
    >>> assert p1 + p2 == Point([5, 7, 9])
    >>> assert p1.dist(p2) ** 2 == 27
    >>> assert (p3.dist(p4) == dist34).all()
    >>> assert p1 != p2
    """

    def __new__(cls, orig_3d=(0, 0, 0)):
        """
        Default constructor. If no arguments are given, the point is initialized to (0, 0, 0).

        Parameters
        ----------
        cls : class
            The class of the object.
        orig_3d : tuple, list, or array_like
            Defaults to 3d origin (0, 0, 0).

        Returns
        -------
        obj : Point
            The origin point of 3d space.

        Examples
        --------
        >>> p1 = Point()
        >>> p1
        Point([0, 0, 0])
        """
        obj = np.asarray(orig_3d).view(cls)
        return obj

    def __repr__(self):
        return f"Point{self.xyz}"

    @property
    def x(self):
        if self.ndim == 1:
            return float(self[0])
        else:
            return np.asarray(self[:, 0])

    @property
    def y(self):
        if self.ndim == 1:
            return float(self[1])
        else:
            return np.asarray(self[:, 1])

    @property
    def z(self: np.ndarray):
        """
        Return
        ------
        z : float
            3rd dimension element. 0 if not defined
        """
        if self.ndim == 1:
            try:
                return float(self[2])
            except IndexError:
                raise IndexError("Point is not 3D.")
        else:
            try:
                return np.asarray(self[:, 2])
            except IndexError:
                raise IndexError("Point is not 3D.")

    @property
    def xyz(self):
        try:
            return np.array((self.x, self.y, self.z)).T
        except IndexError:
            return np.array((self.x, self.y)).T

    @property
    def size(self):
        return self.shape[0]

    @property
    def bounds(self):
        """
        Returns
        -------
        bounds : array_like
            The bounding box of the point. The first row is the minimum
            values and the second row is the maximum values for each
            dimension.
        """
        min = np.min(self.xyz, axis=0)
        max = np.max(self.xyz, axis=0)

        return np.array((min, max))

    def set_x(self, x):
        if self.ndim == 1:
            self[0] = x
        else:
            self[:, 0] = x

    def set_y(self, y):
        if self.ndim == 1:
            self[1] = y
        else:
            self[:, 1] = y

    def set_z(self, z):
        # check if self.z is IndexError
        if self.ndim == 1:
            try:
                self[2] = z
            except IndexError:
                raise IndexError("Point is not 3D, cannot set z.")
        else:
            try:
                self[:, 2] = z
            except IndexError:
                raise IndexError("Point is not 3D, cannot set z.")

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not np.array_equal(self, other)

    def dist(self, other):
        """
        Both points must have the same dimensions
        :return: Euclidean distance
        """
        # check if both array have the same shape
        if self.shape != other.shape:
            raise ValueError("Point datasets must have the same dimensionality and size.")

        if self.ndim == 1:
            return np.linalg.norm(self - other)
        else:
            return np.linalg.norm(self - other, axis=1)

    def direction_ratio(self, other):
        """
        Gives the direction ratio between 2 points
        Parameters
        ----------
        other : Point object
            The other point to which the direction ratio is calculated.

        Returns
        -------
        direction_ratio : list
            The direction ratio between the 2 points.

        Examples
        --------
        >>> from polykriging.geometry import Point
        >>> p1 = Point(1, 2, 3)
        >>> p1.direction_ratio(Point(2, 3, 5))
        [1, 1, 2]
        """
        return [(other.x - self.x), (other.y - self.y), (other.z - self.z)]

    def save_as_vtk(self, filename, color=None):
        """
        Save the point as a vtk file.
        """
        if color is None:
            color = {}
        # check if color is a np.ndarray
        elif isinstance(color, np.ndarray):
            if color.shape[0] != self.shape[0]:
                raise ValueError("Color array must have the same size as the point array.")
        else:
            filename = filename + ".vtk" if not filename.endswith(".vtk") else filename
            ptx.save_ply(filename, vertices=self.xyz,
                        point_data=color, binary=False)


class Vector(Point):
    """
    A vector class inheriting from Point. This class is used to represent
    vectors in n-dimensional space. The shape of the array is (1, n).

    Examples
    --------
    >>> from polykriging.geometry import Vector
    >>> v1 = Vector((1, 2, 3))
    >>> v2 = Vector((4, 5, 7))
    >>> sum12 = Vector((5, 7, 10))
    >>> dot12 = Vector((4, 10, 21))
    >>> assert v1 + v2 == sum12
    >>> assert v1 * v2 == dot12  # Dot product
    """

    # TODO: check if the input is a 1d array. If not, raise an error

    def __repr__(self):
        return "Vector(%g, %g, %g)" % (self.x, self.y, self.z)

    @property
    def add(self, other):
        # it is not necessary to check the shape of the array
        # because numpy will do it automatically.
        return self + other

    @property
    def sub(self, other):
        return self - other

    def dot(self, other):
        return np.dot(self, other)

    def cross(self, other):
        return np.cross(self, other)

    @property
    def norm(self):
        """
        Return
        ------
        norm : float
            The norm of the vector.
        """
        return np.linalg.norm(self)

    def angle_between(self, other, radian=False):
        """
        Return the angle between 2 vectors.

        .. math::
            \\theta = \\arccos \\frac{v_1 \\cdot v_2}{\\|v_1\\| \\|v_2\\|}

        Parameters
        ----------
        other : Vector object
            The other vector to which the angle is calculated.
        radian : bool, optional
            If True, return the angle in radians. The default is False.

        Returns
        -------
        angle : float
            The angle between the 2 vectors. If radian is True, the angle is in radians.
            Otherwise, the angle is in degrees.
        """
        if other is not Vector:
            try:
                other = Vector(other)
            except ValueError:
                raise ValueError("The other object must be a Vector")

        if radian:
            return np.arccos(self.dot(other) / (self.norm * other.norm))
        else:
            return np.arccos(self.dot(other) / (self.norm * other.norm))


class Line(symLine.Line):
    """
    A line class. This class is used to represent a line in n-dimensional space.
    The line is defined by two points: p1 and p2. This is a wrap of sympy.geometry.line.Line.

    So far, please refer to the documentation of sympy.geometry.line.Line for more information.
        https://docs.sympy.org/latest/modules/geometry/lines.html

    TODO : A detailed documentation will be added in the future.

    Examples
    --------
    >>> from polykriging.geometry import Point, Line
    >>> p1 = Point([1, 1, 1])
    >>> p2 = Point([2, 3, 4])
    >>> l1 = Line(p1, p2)
    >>> l1.__repr__()
    'Line(Point((1, 1, 1)), Point((2, 3, 4)))'
    >>> l1.ambient_dimension
    3
    """

    def __new__(cls, p1, p2=None, **kwargs):
        """
        Parameters
        ----------
        p1 : Point object
            The first point of the line.
        p2 : Point object
            The second point of the line.
        """
        if p1 == p2:
            # sometimes we return a single point if we are not given two unique
            # points. This is done in the specific subclass
            raise ValueError(
                "%s.__new__ requires two unique Points." % cls.__name__)
        if len(p1) != len(p2):
            raise ValueError(
                "%s.__new__ requires two Points of equal dimension." % cls.__name__)
        if p1.ndim > 1 or p2.ndim > 1:
            raise ValueError(
                "%s.__new__ requires two Points of 1 dimension." % cls.__name__)

        return GeometryEntity.__new__(cls, p1, p2, **kwargs)

    def __repr__(self):
        return "Line(Point %s, Point %s)" % (self.p1, self.p2)


class Curve:
    """
    This is an open curve defined by a list of points in the order of connection. It is
    created by using composition instead of inheritance of PolyKriging.geometry.Point.

    Examples
    --------
    >>> from polykriging.geometry import Point, Curve
    >>> p = [[2, 4, 6], [2, 4, 5]]
    >>> c = Curve(p)
    >>> c.points
    [[2, 4, 6], [2, 4, 6]]
    >>> c.ambient_dimension
    3
    >>> c.length
    1.0
    >>> c.plot()
    """

    def __init__(self, points):
        """
        A partial inheritance of polykriging.geometry.Point class.

        Parameters
        ----------
        points : list, tuple or array_like
            A list of Point objects.
        """
        self.points = Point(points)
        self.n_points = self.points.shape[0]
        self.dim = self.points.shape[1]
        self.cells = self.__cell_from_points()

        self.__type__ = "Curve"

    @property
    def length(self):
        """
        Return the length of the curve.

        Returns
        -------
        length : float
        """
        return np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))

    @property
    def ambient_dimension(self):
        """
        Return the dimension of the curve.

        Returns
        -------
        ambient_dimension : int
        """
        return self.points.ndim

    @property
    def bounds(self):
        """
        Return the bounds of the curve.

        Returns
        -------
        bounds : tuple
        """
        return self.points.bounds

    @property
    def tangent(self):
        """
        Return the tangent vector of the curve at each point.

        Returns
        -------
        tangent : Vector object
        """
        if self.dim == 2:
            tang = np.zeros((self.points.shape[0], 2))

            x_dist = np.diff(self.points[:, 0]) + 1e-15

            x_prime = np.ones(self.n_points)
            y_prime = [(self.points[i + 1, 1] - self.points[i - 1, 1]) / (x_dist[i - 1] + x_dist[i])
                        for i in range(1, self.n_points - 1)]
            y_prime = np.concatenate(([y_prime[0]], y_prime, [y_prime[-1]]))

            tang[:, 0] = x_prime
            tang[:, 1] = y_prime

            tang = tang / np.linalg.norm(tang, axis=1)[:, None]

        elif self.dim == 3:
            tang = np.zeros((self.points.shape[0], 3))

            x_dist = np.diff(self.points[:, 0]) + 1e-15

            x_prime = np.ones(self.n_points)
            # get y_prime with central difference
            y_prime = [(self.points[i + 1, 1] - self.points[i - 1, 1]) / (x_dist[i - 1] + x_dist[i])
                        for i in range(1, self.n_points - 1)]
            y_prime = np.concatenate(([y_prime[0]], y_prime, [y_prime[-1]]))

            # get z_prime with central difference
            z_prime = [(self.points[i + 1, 2] - self.points[i - 1, 2]) / (x_dist[i - 1] + x_dist[i])
                        for i in range(1, self.n_points - 1)]
            z_prime = np.concatenate(([z_prime[0]], z_prime, [z_prime[-1]]))

            tang[:, 0] = x_prime
            tang[:, 1] = y_prime
            tang[:, 2] = z_prime

            tang = tang / np.linalg.norm(tang, axis=1)[:, None]

        return Vector(tang)

    @property
    def curvature(self):
        """
        Return the curvature of the curve.

        TODO: curvature of a curve

        Returns
        -------
        curvature : float
        """
        return NotImplementedError

    def __cell_from_points(self):
        """
        Given an array of points, make a line set

        Returns
        -------
        cell : vtk.vtkCellArray
            The cell array of the curve. The first column is the number of points
            in each line segment (2 for line segment), and the following columns are
            the indices of the points.
        """
        points = self.points
        cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
        cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
        cells[:, 2] = np.arange(1, len(points), dtype=np.int_)

        return cells

    def save(self, save_path=None):
        """
        Save the curve to a vtk file.

        Parameters
        ----------
        save_path : str
            The path to save the vtk file.

        Returns
        -------
        None
        """
        poly = pv.PolyData()
        poly.points = self.points
        poly.lines = self.__cell_from_points()
        self.__curve = poly

        if save_path is not None:
            path, filename = os.path.split(save_path)
            if not os.path.exists(path):
                os.makedirs(path)
            poly.save(save_path)

        return None

    def plot(self):
        """
        Plot the curve.

        Returns
        -------
        None
        """
        if not hasattr(self, "__curve"):
            self.save()
        self.__curve.plot()
        return None

    def to_polygon(self):
        """
        Convert the curve to a polygon.

        Returns
        -------
        polygon : Polygon object
        """
        if np.any(self.points[0, :] - self.points[-1, :] != 0):
            self.points = np.vstack((self.points, self.points[0, :]))
        return Polygon(self.points)


class Ellipse2D:
    """
    This is an ellipse defined by a center point and two vectors.

    Examples
    --------
    >>> from polytex.geometry import Ellipse2D
    >>> e = Ellipse2D(20, 1, 0.5, [0, 0])
    >>> e.__repr__()
    """

    def __init__(self, n: int, a: float, b: float, center=[0, 0]):
        """
        Parameters
        ----------
        n : int
            The number of points to generate.
        a : float
            The length of the major axis.
        b : float
            The length of the minor axis.
        center : list, tuple or array_like
            The center of the ellipse.
        """
        self.center = center
        self.a = a
        self.b = b
        self.n = n

        self.points = self.elipse_2d()
        self.x = self.points[:, 0]
        self.y = self.points[:, 1]

    def elipse_2d(self) -> np.ndarray:
        """
        Generate points on an ellipse.

        Returns
        -------
        xy: array-like
            Points on the ellipse with shape (n, 2).
        """
        n = self.n
        a = self.a
        b = self.b
        center = self.center

        xy = np.zeros((n, 2))
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)

        pc = np.array(center, dtype=np.float32)
        xy[:, 0] = a * np.cos(theta) + pc[0]
        xy[:, 1] = b * np.sin(theta) + pc[1]
        return xy


class Polygon(Curve):
    """
    This is a closed curve in 2/3-dimensional space. The polygon is defined
    by a list of points. The first and last point must be the same. if not,
    the last point will be added to the list.

    TODO : check if all the points are in the same plane.

    Examples
    --------
    >>> from polytex.geometry import Point, Polygon
    >>> p = Point([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]])
    >>> poly = Polygon(p)
    >>> poly.points
    """

    def __init__(self, points):
        """
        A partial inheritance of polytex.geometry.Point class.

        Parameters
        ----------
        points : list, tuple or array_like
            A list of Point objects.
        """
        if np.any(points[0, :] - points[-1, :] != 0):
            points = np.vstack((points, points[0, :]))

        self.points = Point(points)
        self.n_points = self.points.shape[0]
        self.cells = self._Curve__cell_from_points()

    def to_curve(self):
        """
        Convert the polygon to a curve.

        Returns
        -------
        curve : Curve object
        """
        return Curve(self.points[:-1, :])

    @property
    def centroid(self):
        """
        Return the centroid of the polygon.

        Returns
        -------
        centroid : Point object
        """
        return Point(np.mean(self.points[:-1, :], axis=0))

    @property
    def area(self):
        """
        Return the area of the polygon.

        Returns
        -------
        area : float
        """
        return NotImplementedError

    @property
    def perimeter(cls):
        """
        Return the perimeter of the polygon.

        Returns
        -------
        perimeter : float
        """
        return cls.length


class Plane:
    """
    A plane in 3D space. The plane can be defined by a point and a normal vector,
    or by three points.

    Examples
    --------
    >>> from polytex.geometry import Point, Plane

    Create a plane from 3 points
    >>> p1 = Point([1, 1, 1])
    >>> p2 = Point([2, 3, 4])
    >>> p3 = Point([2, 2, 2])
    >>> plane1 = Plane(p1, p2, p3)

    Create a plane from a point and a normal vector
    >>> p1 = Point([1, 1, 1])
    >>> normal = Point([1, 4, 7])
    >>> plane2 = Plane(p1, normal_vector=normal)
    """

    def __init__(self, p1, a=None, b=None, **kwargs):
        """
        Create a plane from 3 points or a point and a normal vector.

        Parameters
        ----------
        p1 : Point object
            A point on the plane.
        a : Point object, optional
            A point on the plane. The default is None.
        b : Point object, optional
            A point on the plane. The default is None.
        **kwargs : dict, optional
            `normal_vector` can be passed as a keyword argument when leaving
            a and b as None. If both a and b are not None, normal_vector will
            be ignored.

        Returns
        -------
        plane : Plane object
        """
        p1 = Point(p1)

        if a is not None and b is not None:
            p2 = Point(a)
            p3 = Point(b)

            self.p2 = p2
            self.p3 = p3

            # p1.tolist(): convert the point to a list for compatibility with sympy geometry
            # It is originally a PolyKriging Point object which is inherited from numpy.ndarray.
            if symPoint.Point3D.are_collinear(p1.tolist(), p2.tolist(), p3.tolist()):
                raise ValueError('Enter three non-collinear points')
            a = p1.direction_ratio(p2)
            b = p1.direction_ratio(p3)
            normal_vector = tuple(Vector(a).cross(Vector(b)))
        else:
            # kwargs is a dictionary of keyword arguments. It is used to pass the normal vector
            # to the sympy.geometry.plane.Plane class. pop() is used to remove the normal vector
            # from the dictionary and return it.
            a = kwargs.pop('normal_vector', a)

            # get() is used to get the normal vector from the dictionary.
            evaluate = kwargs.get('evaluate', True)

            # check if a is a Point object
            if not isinstance(a, Vector):
                a = Vector(a)

            if not len(a) == 3:
                raise ValueError("""Either provide 3 3D points or a point with a
                    normal vector expressed as a sequence of length 3""")

            normal_vector = a

            if np.all(a == 0):
                raise ValueError('Normal vector cannot be zero vector')

        self.p1 = p1
        self.normal = Vector(np.array(normal_vector / normal_vector.norm, dtype=np.float32))



    def __repr__(self):
        return f"Plane(Point{self.p1}, Normal{self.normal})"

    def distance(self, point):
        """
        Return the signed distance between a point and the plane.

        Parameters
        ----------
        point : Point object
            The point to calculate the distance. shape = (n, 3), where n is the number of points.

        Returns
        -------
        distance : float
            The distance between the point and the plane.

        Examples
        --------
        >>> from polytex.geometry import Point, Plane
        >>> p1 = Point([5, 0, 0])
        >>> normal = Point([1, 0, 0])
        >>> plane = Plane(p1, normal_vector=normal)
        >>> plane.show()
        >>> plane.distance([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        array([-5., -4., -3.])
        """
        normal = self.normal
        p1 = self.p1

        # check if point is a Point object
        if not isinstance(point, Point):
            point = Point(point)

        dist = (normal[0] * (point.x - p1.x)
                + normal[1] * (point.y - p1.y)
                + normal[2] * (point.z - p1.z))
        return dist

    def show(self):
        """
        Plot the plane.

        Returns
        -------
        None
        """
        p = pv.Plane(center=self.p1, direction=self.normal,
                  i_size=1, j_size=1,
                  i_resolution=2, j_resolution=2)
        p.plot(show_edges=False)

    # def __dir__(self):
    #     return []

    def function(self):
        """
        Return the equation of the plane as a function of x, y and z.

        Note
        ----
            normal = (a, b, c)
            point = (x0, y0, z0)
            Equation of a plane: a(x-x0) + b(y-y0) + c(z-z0) = 0

        Parameters
        ----------
        normal : array-like
            normal vector of the plane.
        point : array-like
            point on the plane.

        Returns
        -------
        A lambda function of x, y and z.
            function of plane.

        Example
        -------
        Create a plane from a point and a normal vector
        >>> from polytex.geometry import Point, Plane
        >>> p1 = Point([1, 1, 1])
        >>> normal = Point([1, 4, 7])
        >>> plane2 = Plane(p1, normal_vector=normal)
        >>> f = plane2.function()
        >>> f
        <function polytex.geometry.basic.Plane.function.<locals>.<lambda>(x, y, z)>
        >>> f(1, 1, 1)
        0
        """
        # definition of the plane for intersection with the curve
        normal = np.array(self.normal)
        point = np.array(self.p1)

        f = lambda x, y, z: normal[0] * (x - point[0]) \
                            + normal[1] * (y - point[1]) \
                            + normal[2] * (z - point[2])
        return f

    def intersection(self, obj, max_dist):
        """
        Return the intersection of the plane with a curve or a polygon.

        Parameters
        ----------
        obj : Curve or Polygon object
            The object to intersect with the plane.
        max_dist : float
            The maximum distance of checked points from the plane.

        Returns
        -------
        intersection : list or array
            The intersection point of the plane with the obj in the shape of
            (1, 3). If the intersection is not found, none is returned.
        """
        try:
            obj_type = obj.__type__
        except AttributeError:
            raise TypeError('Object must be a Curve, ParamCurve or ParamSurface '
                            'object defined in polytex.geometry')

        if obj_type == 'Curve':
            f = self.function()
            points = obj.points
            mask = np.linalg.norm(points - self.p1, axis=1) < max_dist

            try:
                pts_intersect = find_intersect(f, points[mask, :])
                return pts_intersect
            except RuntimeError:
                # print('No intersection found')
                return None
        else:
            # TODO: implement intersection with ParamCurve and ParamSurface
            raise NotImplementedError


class Tube(GeometryEntity):
    """
    This class defines a 3D tubular surface by number of points on each cross-section (theta_res)
    and the number of cross-sections (h_res). Note that the number of points on the cross-section
    is the same for all the-cross sections.

    Examples
    --------
    >>> from polytex.geometry import Tube
    >>> tube = Tube(4,10,major=2, minor=1,h=5)
    >>> mesh = tube.mesh(plot=True)
    >>> tube.save_as_mesh('tube.vtk')
    """

    def __new__(cls, theta_res, h_res, vertices=None, **kwargs):
        """
        Parameters
        ----------
        theta_res : int
            The number of points on each cross-section.
        h_res : int
            The number of cross-sections.
        points : array_like
            The points on the cross-sections. The shape of the array should be (h_res * theta_res, 3).
            The points should be ordered in the following way:
                [p1, p2, ..., p_theta_res, p1, p2, ..., p_theta_res, ..., p1, p2, ..., p_theta_res]
            where p1, p2, ..., p_theta_res are the points on each cross-section from the top to the bottom.

            the default value is None. If the value is None, the points will be generated automatically by assigning
            the height, major and minor radius to the tube.

        major : float
            The major radius of the tube. Only used when the points are not given.
        minor : float
            The minor radius of the tube. Only used when the points are not given.
        h : float
            The height of the tube. Only used when the points are not given.
        """

        if vertices is not None:
            vertices = np.array(vertices)
            if vertices.shape[0] != h_res * theta_res:
                raise ValueError('The number of vertices is not equal to the number of cross-sections times the'
                                 ' number of vertices on each cross-section')
            if vertices.shape[1] != 3:
                raise ValueError('The vertices must be in the shape of (n, 3)')
        else:
            cls.a = kwargs.pop('major')
            cls.b = kwargs.pop('minor')
            cls.h = kwargs.pop('h')

        return super().__new__(cls, theta_res, h_res, vertices, **kwargs)

    @property
    def theta_res(self) -> int:
        return self.args[0]

    @property
    def h_res(self) -> int:
        return self.args[1]

    @property
    def points(self):
        if self.args[2] is not None:
            return self.args[2]
        else:
            pts = ms.structured_cylinder_vertices(a=self.a, b=self.b, h=self.h,
                                                  theta_res=self.theta_res, h_res=self.h_res)
            return pts

    @property
    def bounds(self):
        min = np.min(self.points, axis=0)
        max = np.max(self.points, axis=0)

        return [min[0], max[0], min[1], max[1], min[2], max[2]]

    def mesh(self, plot=False, show_edges=True):
        """
        TODO : raise TypeError("Given points must be a sequence or an array.")

        Note
        ----
        theta_res should be 1 less then else where. To be fixed in the future.
        """
        theta_res, h_res = int(self.theta_res), int(self.h_res)
        pts = np.array(self.points, dtype=np.float32)
        pv_mesh = ptx.mesh.tubular_mesh_generator(theta_res=theta_res - 1, h_res=h_res,
                                                 vertices=pts, plot=False)
        if plot:
            pv_mesh.plot(show_edges=True)
        return pv_mesh

    def save_as_mesh(self, save_path, end_closed=True):
        """
        Save the tubular mesh to a file. The file format is determined by the extension of the filename.
        The possible file formats are: [".ply", ".stl", ".vtk", ".vtu"].

        TODO : There seems to be a bug in correction option of the to_meshio_data() method of the tubular mesh.

        Parameters
        ----------
        save_path : str
            The path and the name of the file to be saved with the extension.
        end_closed : bool
            If True, the ends of the tube will be closed. The default value is True.

        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            The tubular mesh.

        Examples
        --------
        >>> from polytex.geometry import Tube
        >>> tube = Tube(5,10,major=2, minor=1,h=5)
        >>> tube.save_as_mesh('tube.vtu')
        """
        mesh = self.mesh()
        # check if path is valid
        if not os.path.exists(os.path.dirname(save_path)):
            # create the directory
            os.makedirs(os.path.dirname(save_path))

        if save_path.endswith('.stl'):
            mesh = mesh.cast_to_unstructured_grid()
            mesh = mesh.triangulate()

        if end_closed:
            points, cells, point_data, cell_data = ms.to_meshio_data(
                mesh,
                int(self.theta_res),
                correction=end_closed)

            ptx.meshio_save(save_path, points, cells=cells,
                           point_data={}, cell_data={}, binary=False)
        else:
            import pyvista as pv
            pv.save_meshio(save_path, mesh, binary=False)

        return mesh


class ParamCurve:
    """
    This is a parametric curve. It is defined by a function and
    a parameter.

        The class is a wrap of sympy.geometry.curve.Curve. So far, please refer to the
    documentation of sympy.geometry.curve.Curve for more information.
        https://docs.sympy.org/latest/modules/geometry/curves.html

    TODO : A detailed documentation will be added in the future.

    Examples
    --------
    >>> from polytex.geometry import ParamCurve
    >>> from sympy import sin, cos, symbols
    >>> import numpy as np
    >>> s = symbols('s')
    >>> curve = ParamCurve(limits=(s, 0, 2*np.pi), function=(cos(s), sin(s)))
    >>> curve
    """

    def __new__(cls, limits, function=[], dataset=None, krig_config=("lin", "cub"),
                smooth=0.0, verbose=False):
        """
        Parameters
        ----------
        limits : 3-tuple
            Function parameter and lower and upper bounds.
        function : list
            The function list for each coordinate component. The default value is [].
        dataset : array_like
            The dataset of the curve. The default value is None. The first column is
            the parameter and the other columns are the value of coordinate components.

            One of the function or dataset must be given. Please note that both are
            given, the dataset will be ignored.
        krig_config : tuple
            The kriging interpolation configuration. The default value is ("lin", "cub").
            The tuple should be in the form of (drift_name, covariance_name).
        smooth : float
            The smoothing factor. The default value is 0.0.
        verbose : bool
            If True, print the information of the kriging process. The default value is False.

        Returns
        -------
        curve : ParamCurve object
        """
        cls.limits = limits

        if len(function) != 0:
            if not is_sequence(function) or len(function) != 2:
                raise ValueError("Function argument should be (x(t), y(t)) "
                                 "but got %s" % str(function))
            if not is_sequence(limits) or len(limits) != 3:
                raise ValueError("Limit argument should be (t, tmin, tmax) "
                                 "but got %s" % str(limits))
            cls.function = function
            return super().__new__(cls)

        if dataset is not None:
            cls.function = []
            dataset = np.array(dataset)
            n_component = dataset.shape[1] - 1

            drift, cov = krig_config
            smoothing_factor = smooth

            for i in range(n_component):
                data_krig = dataset[:, [0, i + 1]]
                if verbose:
                    print("Creating kriging model for %s -th component" % str(i + 1))

                mat_krig, mat_krig_inv, vector_ba, expr, func_drift, func_cov = \
                    ptx.kriging.curve_krig_2D(data_krig, name_drift=drift,
                                           name_cov=cov, nugget_effect=smoothing_factor)

                cls.function.append(expr)
            if verbose:
                print("Kriging model is created successfully for all components.")

            return super().__new__(cls)

    def __repr__(self):
        return "Limits: " + str(self.limits) + "; Functions: " + str(self.function)

    def bounds(self):
        t_value = np.linspace(float(self.limits[1]), float(self.limits[2]), 50)
        print("This is an estimate of the bounds with 50 points. "
              "The actual bounds may be different.")
        curve = self.eval(t_value)
        return curve.bounds

    def eval(self, t_value):
        """
        Evaluate the curve at a given parameter value. The parameter value
        should be within the limits. Otherwise, an error will be raised.

        t_value : float or array_like
            The parameter value for evaluation.
        """
        t, tmin, tmax = self.limits

        # TODO : This is a temporary solution.
        #  The actual parametric variable should be used.
        if isinstance(t, str):
            t = sym.Symbol("x")

        if np.min(t_value) < tmin or np.max(t_value) > tmax:
            raise ValueError("Parameter value should be within the limits.")

        for func in self.function:
            func_x = sym.lambdify(t, func, 'numpy')
            x_t = func_x(np.array(t_value))
            try:
                interpolated_curve = np.vstack((interpolated_curve, x_t))
            except NameError:
                interpolated_curve = x_t

        return Curve(np.array(interpolated_curve).T)


# TODO : An inheritance of sympy.geometry.entity.GeometryEntity should be enough.
class ParamCurve3D(symCurve.Curve):
    """
    This is a parametric curve defined by a function.

    Examples
    --------
    >>> from polytex.geometry import ParamCurve3D
    >>> from sympy import sin, cos, symbols
    >>> s = symbols('s')
    >>> curve = ParamCurve3D((cos(s), sin(s), s), (s, 0, 2*np.pi))
    >>> curve
    """

    def __new__(cls, functions, limits, **kwargs):
        """
        Parameters
        ----------
        function : list of functions
            Function argument should be (x(t), y(t), z(t)) for a 3D curve.
        limits : 3-tuple
            Function parameter and lower and upper bounds. The parameter should be
            the same for all three functions. For example, (t, 0, 1) is valid but
            ((t, 0, 1), (s, 0, 1), (u, 0, 1)) is not.
        """
        if not is_sequence(functions) or len(functions) != 3:
            raise ValueError("Function argument should be (x(t), y(t), z(t)) "
                             "but got %s" % str(functions))
        if not is_sequence(limits) or len(limits) != 3:
            raise ValueError("Limit argument should be (t, tmin, tmax) "
                             "but got %s" % str(limits))

        return GeometryEntity.__new__(cls, Tuple(*functions), Tuple(*limits), **kwargs)

    def eval(self, t_value):
        """
        t : float or array_like
            The parameter value for evaluation.
        """
        t, tmin, tmax = self.limits

        if np.min(t_value) < tmin or np.max(t_value) > tmax:
            raise ValueError("Parameter value should be within the limits.")

        func_x = sym.lambdify(t, self.functions[0], 'numpy')
        x_t = func_x(np.array(t_value))

        func_y = sym.lambdify(self.limits[0], self.functions[1], 'numpy')
        y_t = func_y(np.array(t_value))

        func_z = sym.lambdify(self.limits[0], self.functions[2], 'numpy')
        z_t = func_z(np.array(t_value))

        return Curve(np.array([x_t, y_t, z_t]).T)

    @property
    def length(self):
        """
        The length of the curve.
        """

        raise NotImplementedError("The length of a parametric curve is not "
                                  "implemented yet. Please use the ParamCurve3D.eval() method "
                                  "to evaluate the curve at a given parameter value list and "
                                  "then calculate the length of the returned curve.")

    @property
    def bounds(self):
        t_value = np.linspace(float(self.limits[1]), float(self.limits[2]), 50)
        print("This is an estimate of the bounds with 50 points. "
              "The actual bounds may be different.")
        curve = self.eval(t_value)
        return curve.bounds

    def translate(self, x=0, y=0, z=0):
        """Translate the Curve by (x, y, z).
        Returns
        =======
        Curve :
            returns a translated curve.
        Examples
        ========
        >>> from polytex.geometry import ParamCurve3D
        >>> from sympy.abc import x
        >>> ParamCurve3D((x, x), (x, 0, 1)).translate(1, 2)
        ParamCurve3D((x + 1, x + 2), (x, 0, 1))
        """
        fx, fy, fz = self.functions
        return self.func((fx + x, fy + y, fz + z), self.limits)

    def rotate(self, angle, axis='z'):
        return NotImplementedError("The rotate method is not implemented yet for parametric curve.")

    def scale(self, x=1, y=1, z=1):
        fx, fy, fz = self.functions
        return self.func((fx * x, fy * y, fz * z), self.limits)


class ParamSurface(GeometryEntity):
    """
    A parametric surface is defined by two parametric equations in the form of
        x = f(s, t)
        y = g(s, t)
        z = h(s, t)
    The surface is defined by the domain of s and t.
    store: s,t, x,y,z

    Examples
    --------
    >>> from polytex.geometry import ParamSurface
    >>> from sympy import sin, cos, symbols
    >>> s, t = symbols('s t')
    >>> surface = ParamSurface((cos(s)*cos(t), sin(s)*cos(t), sin(t)), ((s, 0, 2*np.pi), (t, 0, np.pi)))
    >>> surface
    ParamSurface((cos(s)*cos(t), sin(s)*cos(t), sin(t)), ((s, 0, 6.28318530717959), (t, 0, 3.14159265358979)))
    >>> surface.eval(0.5, 0.5)

    """

    def __new__(cls, functions, limits, **kwargs):
        """
        Parameters
        ----------
        functions : list of functions
            Function argument should be (x(s, t), y(s, t), z(s, t)) for a 3D surface.
        limits : 2-tuple
            Function parameter and lower and upper bounds of the two parameters. For
            example, ((s, 0, 1), (t, 0, 1)) is valid. The parameter should be
            the same for all three functions.
        """
        if not is_sequence(functions) or len(functions) != 3:
            raise ValueError("Function argument should be (x(t), y(t), z(t)) "
                             "but got %s" % str(functions))

        cond_limits = is_sequence(limits) and len(limits) == 2 and \
                      is_sequence(limits[0]) and len(limits[0]) == 3 and \
                      is_sequence(limits[1]) and len(limits[1]) == 3
        if not cond_limits:
            raise ValueError("Limit argument should be ((s, smin, smax), (t, tmin, tmax))"
                             "but got %s" % str(limits))

        return GeometryEntity.__new__(cls, Tuple(*functions), Tuple(*limits), **kwargs)

    @property
    def functions(self):
        """The functions specifying the surface.

        Returns
        -------
        functions :
            list of parameterized coordinate functions.

        Examples
        --------
        >>> from sympy.abc import t
        >>> from polytex.geometry import ParamCurve3D
        >>> surface = ParamSurface((t, t, t), ((t, 0, 1), (t, 0, 1)))
        >>> surface.functions
        (t, t, t)
        """
        return self.args[0]

    @property
    def limits(self):
        """The limits of the two parameters specifying the surface.

        Returns
        -------
            list limits of the parameters.

        Examples
        --------
        >>> from sympy.abc import t
        >>> from polytex.geometry import ParamCurve3D
        >>> surface = ParamSurface((t, t, t), ((t, 0, 1), (t, 0, 1)))
        >>> surface.limits
        ((t, 0, 1), (t, 0, 1))
        """
        return self.args[1]

    def eval(self, s_value, t_value):
        """
        Evaluate the surface at the given parameter values.

        Parameters
        ----------
        s_value : float or array_like
            The parameter value for evaluation.
        t_value : float or array_like
            The parameter value for evaluation.

        Returns
        -------
        Surface :
            The surface evaluated at the given parameter values.
            the shape of the returned surface is (len(s) * len(t), 3). The
            first column is s values, the second column is t values, and the
            following columns are the coordinates of the surface at the given
            parameter values (x, y, z).
        """
        s, smin, smax = self.limits[0]
        t, tmin, tmax = self.limits[1]

        if np.min(s_value) < smin or np.max(s_value) > smax:
            raise ValueError("First parameter value should be within the limits.")
        if np.min(t_value) < tmin or np.max(t_value) > tmax:
            raise ValueError("Second parameter value should be within the limits.")

        sv, tv = np.meshgrid(s_value, t_value)
        sv, tv = (sv.flatten()).T, (tv.flatten()).T

        func_x = sym.lambdify((s, t), self.functions[0], 'numpy')
        x_st = func_x(np.array(sv), np.array(tv))

        func_y = sym.lambdify((s, t), self.functions[1], 'numpy')
        y_st = func_y(np.array(sv), np.array(tv))

        func_z = sym.lambdify((s, t), self.functions[2], 'numpy')
        z_st = func_z(np.array(sv), np.array(tv))

        return np.array([sv, tv, x_st, y_st, z_st]).T

    def translate(self, x=0, y=0, z=0):
        fx, fy, fz = self.functions
        return self.func((fx + x, fy + y, fz + z), self.limits)

    def rotate(self, angle, axis='z'):
        return NotImplementedError("The rotate method is not implemented yet for parametric surface.")

    def scale(self, x=1, y=1, z=1):
        fx, fy, fz = self.functions
        return self.func((fx * x, fy * y, fz * z), self.limits)


def find_intersect(f, curve, niterations=5, mSegments=5):
    """
    Find the intersection of a curve with a plane

    Parameters
    ----------
    f : lambda function
        function of plane.
    curve : array-like
        points on the curve in shape of (n, 3).
    niterations: int
        number of iterations.
    mSegments: int
        number of segments for each iteration.

    Returns
    -------
    intersection: array-like
        intersection points with shape (n, 3).
    """
    # check if curve is a numpy array
    if not isinstance(curve, np.ndarray):
        curve = np.array(curve)
    if curve.shape[1] != 3:
        raise RuntimeError("The shape of curve must be (n, 3).")

    fSign = f(curve[:, 0], curve[:, 1], curve[:, 2])
    idx = np.where(np.diff(np.sign(fSign)))[0]

    # if there is no intersection
    if len(idx) == 0:
        raise RuntimeError("No intersection was detected.")
    elif len(idx) > 1:
        print("Warning: More than one intersection was detected. "
              "However, only the first one is handled.")

    # TODO: cases with more than one intersection
    while niterations > 0:
        # refine the result by linear interpolation
        niterations -= 1
        low = curve[idx, :]
        high = curve[idx + 1, :]
        curve = np.squeeze(
            np.linspace(low, high, mSegments).reshape(1, -1, 3))
        fSign = f(curve[:, 0], curve[:, 1], curve[:, 2])
        idx = np.where(np.diff(np.sign(fSign)))[0]

    intersect = (curve[idx, :] + curve[idx + 1, :]) / 2

    return intersect


if __name__ == "__main__":
    import doctest

    doctest.testmod()
