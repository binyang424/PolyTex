from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import eye
from sympy.printing import sstr
from sympy.utilities.iterables import is_sequence


class GeometryEntity(Basic, EvalfMixin):
    """The base class for all geometrical entities.

    This class does not represent any particular geometric entity, it only
    provides the implementation of some methods common to all subclasses.
    """

    # __slots__ = ()

    def __contains__(self, other):
        """Subclasses should implement this method for anything more complex than equality."""
        if type(self) is type(other):
            return self == other
        raise NotImplementedError()

    def __getnewargs__(self):
        """Returns a tuple that will be passed to __new__ on unpickling."""
        return tuple(self.args)

    def __new__(cls, *args, **kwargs):
        # Points are sequences, but they should not
        # be converted to Tuples, so use this detection function instead.
        def is_seq_and_not_point(a):
            # we cannot use isinstance(a, Point) since we cannot import Point
            if hasattr(a, 'is_Point') and a.is_Point:
                return False
            return is_sequence(a)

        args = [Tuple(*a) if is_seq_and_not_point(a) else sympify(a) for a in args]
        return Basic.__new__(cls, *args)

    def __repr__(self):
        """String representation of a GeometryEntity that can be evaluated
        by sympy."""
        return type(self).__name__ + repr(self.args)

    def __str__(self):
        """String representation of a GeometryEntity."""
        return type(self).__name__ + sstr(self.args)

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.
        """
        raise NotImplementedError()

    def encloses(self, o):
        """
        Return True if o is inside (not on or outside) the boundaries of self.

        The object will be decomposed into Points and individual Entities need
        only define an encloses_point method for their class.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point
        sympy.geometry.polygon.Polygon.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t  = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t2 = Polygon(*RegularPolygon(Point(0, 0), 2, 3).vertices)
        >>> t2.encloses(t)
        True
        >>> t.encloses(t2)
        False
        """

        from sympy.geometry.point import Point
        from sympy.geometry.line import Segment, Ray, Line
        from sympy.geometry.ellipse import Ellipse
        from sympy.geometry.polygon import Polygon, RegularPolygon

        if isinstance(o, Point):
            return self.encloses_point(o)
        elif isinstance(o, Segment):
            return all(self.encloses_point(x) for x in o.points)
        elif isinstance(o, (Ray, Line)):
            return False
        elif isinstance(o, Ellipse):
            return self.encloses_point(o.center) and \
                self.encloses_point(
                Point(o.center.x + o.hradius, o.center.y)) and \
                not self.intersection(o)
        elif isinstance(o, Polygon):
            if isinstance(o, RegularPolygon):
                if not self.encloses_point(o.center):
                    return False
            return all(self.encloses_point(v) for v in o.vertices)
        raise NotImplementedError()

    def intersection(self, o):
        """
        Returns a list of all of the intersections of self with o.

        Notes
        =====

        An entity is not required to implement this method.

        If two different types of entities can intersect, the item with
        higher index in ordering_of_classes should implement
        intersections with anything having a lower index.

        See Also
        ========

        sympy.geometry.util.intersection

        """
        raise NotImplementedError()

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        The default pt is the origin, Point(0, 0)

        See Also
        ========

        scale, translate

        Examples
        ========

        >>> from sympy import Point, RegularPolygon, Polygon, pi
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t # vertex on x axis
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.rotate(pi/2) # vertex on y axis now
        Triangle(Point2D(0, 1), Point2D(-sqrt(3)/2, -1/2), Point2D(sqrt(3)/2, -1/2))

        """
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):
                newargs.append(a.rotate(angle, pt))
            else:
                newargs.append(a)
        return type(self)(*newargs)

    def scale(self, x=1, y=1, pt=None):
        """Scale the object by multiplying the x,y-coordinates by x and y.

        If pt is given, the scaling is done relative to that point; the
        object is shifted by -pt, scaled, and shifted by pt.

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.scale(2)
        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)/2), Point2D(-1, -sqrt(3)/2))
        >>> t.scale(2, 2)
        Triangle(Point2D(2, 0), Point2D(-1, sqrt(3)), Point2D(-1, -sqrt(3)))

        """
        from sympy.geometry.point import Point
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return type(self)(*[a.scale(x, y) for a in self.args])  # if this fails, override this class

    def translate(self, x=0, y=0):
        """Shift the object by adding to the x,y-coordinates the values x and y.

        See Also
        ========

        rotate, scale

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t
        Triangle(Point2D(1, 0), Point2D(-1/2, sqrt(3)/2), Point2D(-1/2, -sqrt(3)/2))
        >>> t.translate(2)
        Triangle(Point2D(3, 0), Point2D(3/2, sqrt(3)/2), Point2D(3/2, -sqrt(3)/2))
        >>> t.translate(2, 2)
        Triangle(Point2D(3, 2), Point2D(3/2, sqrt(3)/2 + 2), Point2D(3/2, 2 - sqrt(3)/2))

        """
        newargs = []
        for a in self.args:
            if isinstance(a, GeometryEntity):
                newargs.append(a.translate(x, y))
            else:
                newargs.append(a)
        return self.func(*newargs)


def translate(x, y):
    """Return the matrix to translate a 2-D point by x and y."""
    rv = eye(3)
    rv[2, 0] = x
    rv[2, 1] = y
    return rv


def scale(x, y, pt=None):
    """Return the matrix to multiply a 2-D point's coordinates by x and y.

    If pt is given, the scaling is done relative to that point."""
    rv = eye(3)
    rv[0, 0] = x
    rv[1, 1] = y
    if pt:
        from sympy.geometry.point import Point
        pt = Point(pt, dim=2)
        tr1 = translate(*(-pt).args)
        tr2 = translate(*pt.args)
        return tr1*rv*tr2
    return rv


def rotate(th):
    """Return the matrix to rotate a 2-D point about the origin by ``angle``.

    The angle is measured in radians. To Point a point about a point other
    then the origin, translate the Point, do the rotation, and
    translate it back:

    >>> from sympy.geometry.entity import rotate, translate
    >>> from sympy import Point, pi
    >>> rot_about_11 = translate(-1, -1)*rotate(pi/2)*translate(1, 1)
    >>> Point(1, 1).transform(rot_about_11)
    Point2D(1, 1)
    >>> Point(0, 0).transform(rot_about_11)
    Point2D(2, 0)
    """
    s = sin(th)
    rv = eye(3)*cos(th)
    rv[0, 1] = s
    rv[1, 0] = -s
    rv[2, 2] = 1
    return rv