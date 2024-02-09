import numpy as np
from numpy import cos, sin
from .basic import Vector


def rx(phi: float) -> np.ndarray:
    """
    Single axis frame rotation about the X-axis.

    Parameters
    ----------
    phi : float
        The angle between z-axis (or y-axis) of the initial and final frames in radian.
        The rotation is positive if the frame rotates in the counter-clockwise direction
        when viewed from the positive end of x-axis.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    return np.array([[1, 0, 0],
                     [0, cos(phi), sin(phi)],
                     [0, -sin(phi), cos(phi)]])


def ry(theta: float) -> np.ndarray:
    """
    Single axis frame rotation about the Y-axis.

    Parameters
    ----------
    theta : float
        The angle between z-axis (or x-axis) of the initial and final frames in radian.
        The rotation is positive if the frame rotates in the counter-clockwise direction
        when viewed from the positive end of y-axis.
    """
    return np.array([[cos(theta), 0, -sin(theta)],
                     [0, 1, 0],
                     [sin(theta), 0, cos(theta)]])


def rz(psi: float) -> np.ndarray:
    """
    Single axis frame rotation about the Z-axis.

    Parameters
    ----------
    psi : float
        The angle between x-axis (or y-axis) of the initial and final frames in radian.
        The rotation is positive if the frame rotates in the counter-clockwise direction
        when viewed from the positive end of z axis.

    Returns
    -------
    numpy.ndarray
        Rotation matrix.
    """
    return np.array([[cos(psi), sin(psi), 0],
                     [-sin(psi), cos(psi), 0],
                     [0, 0, 1]])


def d2r(degrees: float) -> float:
    """
    Convert degrees to radians.

    Parameters
    ----------
    degrees : float
        Angle in degrees.

    Returns
    -------
    float
        Angle in radians.
    """
    return degrees * (np.pi / 180)


# ====================================================
# e123 rotation sequence
# Note: The code below is from:
# How to Transform a Reference Frame in Python Using NumPy | by Andrew Joseph Davies
# https://python.plainenglish.io/reference-frame-transformations-in-python-with-numpy-and-matplotlib-6adeb901e0b0

def __q11(psi: float, theta: float) -> float:
    return np.cos(psi) * np.cos(theta)


def __q12(psi: float, theta: float, phi: float) -> float:
    return np.cos(psi) * np.sin(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi)


def __q13(psi: float, theta: float, phi: float) -> float:
    return -np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)


def __q21(psi: float, theta: float) -> float:
    return - np.sin(psi) * np.cos(theta)


def __q22(psi: float, theta: float, phi: float) -> float:
    return -np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi)


def __q23(psi: float, theta: float, phi: float) -> float:
    return np.sin(psi) * np.sin(theta) * np.cos(phi) + np.cos(psi) * np.sin(phi)


def __q31(theta: float) -> float:
    return np.sin(theta)


def __q32(theta: float, phi: float) -> float:
    return - np.cos(theta) * np.sin(phi)


def __q33(theta: float, phi: float) -> float:
    return np.cos(theta) * np.cos(phi)


def e123_dcm(psi: float, theta: float, phi: float) -> np.ndarray:
    """
    This function chaining the rotation matrices for the Euler 123 sequence. The
    rotation matrix is defined as:

    .. math::
        R = R_3(psi) R_2(theta) R_1(phi)

    where :math:`R_1` is the rotation matrix about the x-axis, :math:`R_2` is the
    rotation matrix about the y-axis, and :math:`R_3` is the rotation matrix about
    the z-axis.

    Parameters
    ----------
    psi : float
        The rotation angle about the z-axis in radians.
    theta : float
        The rotation angle about the y-axis in radians.
    phi : float
        The rotation angle about the x-axis in radians.

    Returns
    -------
    dcm : numpy.ndarray
        The direction cosine matrix for the Euler 123 sequence.

    Examples
    --------
    >>> import polykriging.geometry.transform as tf

    """
    return np.array([[__q11(psi, theta), __q12(psi, theta, phi), __q13(psi, theta, phi)],
                     [__q21(psi, theta), __q22(psi, theta, phi), __q23(psi, theta, phi)],
                     [__q31(theta), __q32(theta, phi), __q33(theta, phi)]])


# Above code is from:
# How to Transform a Reference Frame in Python Using NumPy | by Andrew Joseph Davies
# https://python.plainenglish.io/reference-frame-transformations-in-python-with-numpy-and-matplotlib-6adeb901e0b0
# ====================================================


def euler_z_noraml(normal, *args) -> list:
    """
    This function returns the euler angles (phi, theta, psi) for
    rotating the global coordinate system to align its z-axis with
    a normal vector from the origin to a point (namely, no translation
    is considered).

    Note
    ----
    No translation is considered. The origin of the global coordinate
    system is assumed to be the origin of the local coordinate system.
    The user should translate the local coordinate system to the origin
    before calling this function and then re-translate the local
    coordinate system to the desired location.

    Parameters
    ----------
    normal : list or array
        The normal vector from the origin to a point.

    Returns
    -------
    euler_angles : list
        The euler angles (psi, theta, phi), where psi is the rotation
        angle about the z-axis, theta is the rotation angle about the
        y-axis, and phi is the rotation angle about the x-axis in radians.
        Note that the rotation should be performed in the order of e123
        by pre-multiplying the rotation matrices.

    Examples
    --------
    >>> import polykriging.geometry.transform as tf
    >>> import numpy as np
    >>> normal = [0.43583834, -0.00777955, -0.89999134]
    >>> euler_angles = tf.euler_z_noraml(normal)
    >>> print(np.allclose(euler_angles, [0, 0.4509695318910846, 3.132948841252596]))
    True
    >>> print("As we are rotating the global coordinate system to align its z-axis with the normal vector,")
    >>> print("the normal vector should be [0, 0, 1] after the rotation.")
    >>> tf.e123_dcm(*euler_angles) @ normal
    >>> print(np.allclose(tf.e123_dcm(*euler_angles) @ normal, [0, 0, 1]))
    True
    """
    z_basis = [0, 0, 1]
    normal = Vector(normal)

    # First, we project the normal vector onto the yz-plane
    normal_proj = normal.copy()
    normal_proj[0] = 0

    # Second, we find the angle between the projected normal vector and the
    # z-axis of the global coordinate system.
    phi = normal_proj.angle_between(z_basis) * (-np.sign(normal_proj[1]))

    normal_proj = Vector(rx(phi).dot(normal_proj))
    normal = Vector(rx(phi).dot(normal))

    # Third, we find the angle between the projected normal vector and the
    # y-axis of the global coordinate system.
    theta = normal.angle_between(normal_proj) * (np.sign(normal[0]))

    return [0, theta, phi]


def euler_zx_coordinate(z_new, x_new) -> list:
    """
    This function returns the euler angles (phi, theta, psi) for
    rotating the global coordinate system to align its z-axis with
    the z_new vector and its x-axis with the x_new vector.

    Note
    ----
    No translation is considered. The origin of the global coordinate
    system is assumed to be the origin of the local coordinate system.
    The user should translate the local coordinate system to the origin
    before calling this function and then re-translate the local
    coordinate system to the desired location.

    Parameters
    ----------
    z_new : list or array
        The coordinate of the new z-axis in the original coordinate system.
    x_new : list or array
        The coordinate of the new x-axis in the original coordinate system.

    Returns
    -------
    euler_angles : list
        The euler angles (psi, theta, phi), where psi is the rotation
        angle about the z-axis, theta is the rotation angle about the
        y-axis, and phi is the rotation angle about the x-axis in radians.
        Note that the rotation should be done in e123 sequence by pre-multiplying
        the rotation matrices.

    Examples
    --------
    >>> import polykriging.geometry.transform as tf
    """
    x_basis = Vector([1, 0, 0])

    angles = euler_z_noraml(z_new)
    temp = np.dot(e123_dcm(*angles), np.vstack([x_new, z_new]).T).T

    # Check the result
    if not np.allclose(temp[1, :], [0, 0, 1]):
        print("Result checking shows the z-axis is not aligned with the normal vector.")

    x_temp = Vector(temp[0, :])
    psi = x_basis.angle_between(x_temp) * np.sign(x_temp[1])
    angles = [psi, angles[1], angles[2]]

    return angles


if __name__ == '__main__':
    v0 = [np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3]  # initial vector

    rm_1 = rx(d2r(-45))  # rotation matrix about x axis
    rm_2 = ry(d2r(35.26439))  # rotation matrix about y-axis

    # Single axis transformation about x-axis followed by y-axis
    v1 = np.dot(rm_1, v0)
    v2_1 = np.dot(rm_2, v1)

    # multiple dot product
    v2_2 = np.linalg.multi_dot([rm_2, rm_1, v0])

    # Chaining multiple rotation matrices
    v2_3 = np.dot(e123_dcm(0, d2r(35.26439), d2r(-45)), v0)

    # test if v2_1, v2_2, v2_3 are equal. We use np.allclose() to
    # compare two arrays instead of == to avoid floating point errors.
    print(np.allclose(v2_1, v2_2) & np.allclose(v2_2, v2_3))
