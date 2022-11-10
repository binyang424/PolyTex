import numpy as np
import sympy as sym
import sympy.geometry.point as symPoint
import sympy.geometry.line as symLine
import sympy.geometry.curve as symCurve
import sympy.geometry.plane as symPlane
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence
from sympy.core.containers import Tuple


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
    >>> from polykriging.geometry import Point
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
        orig_3d : tuple
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
            return self[0]
        else:
            return self[:, 0]

    @property
    def y(self):
        if self.ndim == 1:
            return self[1]
        else:
            return self[:, 1]

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
                return self[2]
            except IndexError:
                raise IndexError("Point is not 3D.")
        else:
            try:
                return self[:, 2]
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


class Vector(Point):
    """
    A vector class inheriting from Point. This class is used to represent
    vectors in n-dimensional space. The shape of the array is (1, n).

    Examples
    --------
    >>> from polykriging.geometry import Vector
    >>> v1 = Vector([1, 2, 3])
    >>> v2 = Vector([4, 5, 7])
    >>> sum12 = Vector([5, 7, 10])
    >>> dot12 = Vector([4, 10, 21])
    >>> dist34 = np.linalg.norm(p3 - p4, axis=1)
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
    >>> p = Point([[2, 4, 6], [2, 4, 5]])
    >>> c = Curve(p)
    >>> c.points
    [[2, 4, 6], [2, 4, 6]]
    >>> c.ambient_dimension
    3
    >>> c.length
    1.0
    """

    def __init__(self, points):
        """
        A partial inheritance of polykriging.geometry.Point class.

        Parameters
        ----------
        points : list
            A list of Point objects.
        """
        self.points = Point(points)

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

        TODO : check and improve the tangent vector calculation.

        Returns
        -------
        tangent : Vector object
        """
        # return np.diff(self.points, axis=0)
        return NotImplementedError

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

    @property
    def ambient_dimension(self):
        """
        Return the dimension of the curve.

        Returns
        -------
        ambient_dimension : int
        """
        return self.points.shape[1]

    def __to_polygon(self):
        """
        Convert the curve to a polygon.

        Returns
        -------
        polygon : Polygon object
        """
        if np.any(self.points[0, :] - self.points[-1, :]):
            self.points = np.vstack((self.points, self.points[0, :]))
        return Polygon(self.points)


class Elipse:
    """
    This is an elipse defined by a center point and two vectors.
    TODO : a wrap of sympy.geometry.ellipse (2D function)
    """

    pass


class Polygon(Curve):
    """
    This is a closed curve in 2/3-dimensional space. The polygon is defined
    by a list of points. The first and last point must be the same. if not,
    the last point will be added to the list.

    TODO : check if all the points are in the same plane.

    Examples
    --------
    >>> from polykriging.geometry import Point, Polygon
    >>> p = Point([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]])
    >>> poly = Polygon(p)
    >>> poly.points
    """

    def to_curve(self):
        """
        Convert the polygon to a curve.

        Returns
        -------
        curve : Curve object
        """
        return Curve(self.points[:-1, :])

    @property
    def centroild(self):
        """
        Return the centroild of the polygon.

        Returns
        -------
        centroild : Point object
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


class Plane(symPlane.Plane):
    """
    This is a wrap of sympy.geometry.plane.Plane class to define a plane in 3D space.
    It can be defined by a point and a normal vector or by 3 points.

    For now, you could refer to the sympy.geometry.plane.Plane class for more information:
        https://docs.sympy.org/latest/modules/geometry/plane.html

    TODO : A detailed documentation will be added later.

    Examples
    --------
    >>> from polykriging.geometry import Point, Plane
    >>> # # create a plane from 3 points
    >>> p1 = Point([1, 1, 1])
    >>> p2 = Point([2, 3, 4])
    >>> p3 = Point([2, 2, 2])
    >>> plane1 = Plane(p1, p2, p3)
    >>> # create a plane from a point and a normal vector
    >>> p1 = Point([1, 1, 1])
    >>> normal = Point([1, 4, 7])
    >>> plane2 = Plane(p1, normal_vector=normal)
    """

    def __new__(cls, p1, a=None, b=None, **kwargs):
        p1 = Point(p1)
        if a is not None and b is not None:
            p2 = Point(a)
            p3 = Point(b)

            # p1.tolist(): convert the point to a list for capatibility with sympy geometry
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
            if is_sequence(a) and len(a) == 3:
                normal_vector = symPoint.Point3D(a).args if evaluate else a
            else:
                raise ValueError('''Either provide 3 3D points or a point with a
                    normal vector expressed as a sequence of length 3''')
            if all(coord.is_zero for coord in normal_vector):
                raise ValueError('Normal vector cannot be zero vector')
        return GeometryEntity.__new__(cls, p1, normal_vector, **kwargs)

    def __repr__(self):
        return f"Plane(Point{self.p1}, Normal{self.normal_vector})"


class ParametricGeometry:
    """
    This is a parametric geometry defined by a function. It can be a curve or
    a surface. The former is a 1D function with 1 parameter and the latter is
    a 2D function with 2 parameters.

    TODO
    """

    pass


class ParamCurve2D(symCurve.Curve):
    """
    This is a parametric curve in 2D space. It is defined by a function and
    a parameter.

        The class is a wrap of sympy.geometry.curve.Curve. So far, please refer to the
    documentation of sympy.geometry.curve.Curve for more information.
        https://docs.sympy.org/latest/modules/geometry/curves.html

    TODO : A detailed documentation will be added in the future.

    Parameters
    ----------
    function : list of functions
    limits : 3-tuple
        Function parameter and lower and upper bounds.

    Examples
    --------
    >>> from polykriging.geometry import ParamCurve2D
    >>> from sympy import sin, cos, symbols
    >>> s = symbols('s')
    >>> curve = ParamCurve2D((cos(s), sin(s)), (s, 0, 2*np.pi))
    >>> curve
    ParametricCurve2D([cos(s), sin(s)], (s, 0, 6.28318530717959))
    """

    def __new__(cls, function, limits):
        if not is_sequence(function) or len(function) != 2:
            raise ValueError("Function argument should be (x(t), y(t)) "
                             "but got %s" % str(function))
        if not is_sequence(limits) or len(limits) != 3:
            raise ValueError("Limit argument should be (t, tmin, tmax) "
                             "but got %s" % str(limits))

        return GeometryEntity.__new__(cls, Tuple(*function), Tuple(*limits))

    def eval(self, t_value):
        """
        Evaluate the curve at a given parameter value. The parameter value
        should be within the limits. Otherwise, an error will be raised.

        t_value : float or array_like
            The parameter value for evaluation.
        """
        t, tmin, tmax = self.limits

        if np.min(t_value) < tmin or np.max(t_value) > tmax:
            raise ValueError("Parameter value should be within the limits.")

        func_x = sym.lambdify(t, self.functions[0], 'numpy')
        x_t = func_x(np.array(t_value))

        func_y = sym.lambdify(self.limits[0], self.functions[1], 'numpy')
        y_t = func_y(np.array(t_value))

        return Curve(np.array([x_t, y_t]).T)

    @property
    def bounds(self):
        t_value = np.linspace(float(self.limits[1]), float(self.limits[2]), 50)
        print("This is an estimate of the bounds with 50 points. "
              "The actual bounds may be different.")
        curve = self.eval(t_value)
        return curve.bounds


# TODO : An inheritance of sympy.geometry.entity.GeometryEntity should be enough.
class ParamCurve3D(symCurve.Curve):
    """
    This is a parametric curve defined by a function.

    Parameters
    ----------
    function : list of functions
        Function argument should be (x(t), y(t), z(t)) for a 3D curve.
    limits : 3-tuple
        Function parameter and lower and upper bounds. The parameter should be
        the same for all three functions. For example, (t, 0, 1) is valid but
        ((t, 0, 1), (s, 0, 1), (u, 0, 1)) is not.

    Examples
    --------
    >>> from polykriging.geometry.basic import ParamCurve3D
    >>> from sympy import sin, cos, symbols
    >>> s = symbols('s')
    >>> curve = ParamCurve3D((cos(s), sin(s), s), (s, 0, 2*np.pi))
    >>> curve
    """
    def __new__(cls, functions, limits,**kwargs):
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
        >>> from polykriging.geometry.basic import ParamCurve3D
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

    Parameters
    ----------
    functions : list of functions
        Function argument should be (x(s, t), y(s, t), z(s, t)) for a 3D surface.
    limits : 2-tuple
        Function parameter and lower and upper bounds of the two parameters. For
        example, ((s, 0, 1), (t, 0, 1)) is valid. The parameter should be
        the same for all three functions.

    Examples
    --------
    >>> from polykriging.geometry.basic import ParamSurface
    >>> from sympy import sin, cos, symbols
    >>> s, t = symbols('s t')
    >>> surface = ParamSurface((cos(s)*cos(t), sin(s)*cos(t), sin(t)), ((s, 0, 2*np.pi), (t, 0, np.pi)))
    >>> surface
    ParamSurface((cos(s)*cos(t), sin(s)*cos(t), sin(t)), ((s, 0, 6.28318530717959), (t, 0, 3.14159265358979)))
    >>> surface.eval(0.5, 0.5)

    """
    def __new__(cls, functions, limits, **kwargs):
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
        >>> from polykriging.geometry.basic import ParamCurve3D
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
        >>> from polykriging.geometry.basic import ParamCurve3D
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
        s : float or array_like
            The parameter value for evaluation.
        t : float or array_like
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

if __name__ == "__main__":
    import doctest

    doctest.testmod()
