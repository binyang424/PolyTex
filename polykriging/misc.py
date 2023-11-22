import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R

def gebart(vf, rf, packing="Quad", tensorial=False):
    """
    Calculate the fiber tow permeability for a given fiber packing
    pattern according to Gebart's model.

    Parameters
    ----------
    vf : float or array_like
        Fiber volume fraction.
    rf : float or array_like
        Fiber radius (m). If vf and rf are arrays, they must have the same
        shape.
    packing : string
        Fiber packing pattern. Valid options are "Quad" and "Hex".
    tensorial : bool
        If True, return the permeability tensor (a list with 9 elements).
        Otherwise, return the permeability components that parallel and
        perpendicular to the fiber tow as a list of 3 floats. The default
        is False.

    Returns
    -------
    k : array-like
        Fiber tow permeability. If tensorial is True, return a list with 9
        elements. Otherwise, return a list of 3 floats (k11, k22, k33),
        corresponding to the permeability components that parallel and
        perpendicular to the fiber tow. The units are m^2. Note that the
        principal permeability components k22 and k33 are equal.

    References
    ----------
    Gebart BR. Permeability of Unidirectional Reinforcements for RTM. Journal of Composite Materials. 1992;26(8):1100-33.

    Examples
    --------
    >>> print(gebart(vf=0.5, rf=1.7e-5, packing="Quad", tensorial=False))
    [2.02807018e-11 3.73472833e-12 3.73472833e-12]
    >>> print(gebart(vf=0.5, rf=1.7e-5, packing="Quad", tensorial=True))
    [2.02807018e-11 0.00000000e+00 0.00000000e+00 0.00000000e+00
     3.73472833e-12 0.00000000e+00 0.00000000e+00 0.00000000e+00
     3.73472833e-12]
    >>> print(gebart(vf=0.5, rf=1.7e-5, packing="Hex", tensorial=False))
    [2.18113208e-11 4.72786599e-12 4.72786599e-12]
    >>> print(gebart(vf=0.5, rf=1.7e-5, packing="Hex", tensorial=True))
    [2.18113208e-11 0.00000000e+00 0.00000000e+00 0.00000000e+00
     4.72786599e-12 0.00000000e+00 0.00000000e+00 0.00000000e+00
     4.72786599e-12]
    """

    # Check input
    if packing not in ["Quad", "Hex"]:
        raise ValueError("Invalid packing pattern. Valid options are 'Quad' and 'Hex'.")

    vf = np.array(vf)
    if np.any(vf <= 0) or np.any(vf >= 1):
        raise ValueError("Fiber volume fraction must be between 0 and 1.")

    rf = np.array(rf)
    if np.any(rf <= 0):
        raise ValueError("Fiber radius must be positive.")

    # Calculate permeability
    if packing == "Quad":
        c1 = 16 / 9 / np.pi / np.sqrt(2)
        vf_max = np.pi / 4
        c = 57
    elif packing == "Hex":
        c1 = 16 / 9 / np.pi / np.sqrt(6)
        vf_max = np.pi / 2 / np.sqrt(3)
        c = 53

    k1 = 8 * rf ** 2 / c * (1 - vf) ** 3 / vf ** 2
    k2 = c1 * rf ** 2 * np.sqrt((np.sqrt(vf_max / vf) - 1) ** 5)

    if tensorial:
        try:
            k = np.zeros([k1.shape[0], 9])
            k[:, 0] = k1
            k[:, 4] = k2
            k[:, 8] = k2
            k = k.tolist()
        except:
            k = [k1, 0, 0, 0, k2, 0, 0, 0, k2]
    else:
        k = [k1, k2, k2]

    return np.array(k)


def porosity_tow(rho_lin, area_xs, rho_fiber=2550, fvf=False):
    """
    Calculate local porosity of a tow based on its cross-sectional area and linear density.

    Parameters
    ----------
    rho_lin: float
        Linear density of the tow. Unit: Tex (g/1000m)
    area_xs: array-like
        Cross-sectional area of the tow. Unit: m^2. Shape: (n cross-sections, 1).
    rho_fiber: float
        Volume density of the fiber. Unit: kg/m^3. Default: 2550 (glass fiber).
    fvf: bool
        Whether to return fiber volume fraction. Default: False. If True, return
        fiber volume fraction instead of porosity.

    Returns
    -------
    porosity: array-like
        Local porosity of the tow. Shape: (n cross-sections, 1). Unit: 1.
        The fiber volume fraction is returned if fvf is True.

    Examples
    --------
    >>> rho_lin = 275 # 275 Tex
    >>> area_xs = np.array([0.16, 0.22, 0.15])/1e6 # mm^2 to m^2
    >>> rho_fiber = 2550 # kg/m^3
    >>> porosity = porosity_tow(rho_lin, area_xs, rho_fiber, fvf=True)
    >>> print(porosity)
    [0.67401961 0.49019608 0.71895425]
    """
    if not isinstance(area_xs, np.ndarray):
        area_xs = np.array(area_xs)

    vf = (rho_lin / 1000 / rho_fiber) / (area_xs * 1000 + 1e-22)

    if np.any(vf < 0.1):
        n = np.sum(vf < 0.9)
        percent = n / vf.shape[0] * 100
        print("Warning: %d (%.2f%%) cross-sections have fiber volume "
              "fraction < 0.1." % (n, percent))

    if fvf:
        return vf

    return 1 - vf


def perm_rotation(permeability, orientation, inverse=False):
    """
    Rotate the permeability tensor according to the yarn
    orientation in the world coordinate system.

    Parameters
    ----------
    permeability: ndarray
        The principal permeability tensor of the yarn in the
        local coordinate system of the yarn. Shape: (n, 9)
    orientation: ndarray
        The orientation of the yarn in the world coordinate
        system. Shape: (n, 3)
    inverse: bool
        If True, the inverse of permeability tensor is returned.

    Returns
    -------
    perm_rot: ndarray
        The rotated permeability tensor. Shape: (n, 9)
    """
    # import tqdm.auto as tqdm

    if not isinstance(permeability, np.ndarray):
        permeability = np.array(permeability)
        assert permeability.shape[1] == 9, "The shape of permeability should be (n, 9)."

    if not isinstance(orientation, np.ndarray):
        orientation = np.array(orientation)
        assert orientation.shape[1] == 3, "The shape of orientation should be (n, 3)."

    D = np.zeros_like(permeability)
    permeability_loc = np.zeros_like(permeability)

    for i in tqdm(range(permeability.shape[0])):
        perm = np.reshape(permeability[i], (3, 3))
        ori = orientation[i]
        r = R.align_vectors([ori], [[1, 0, 0]])[0]
        A = r.as_matrix()
        k_rot = np.dot(np.dot(A, perm), A.T)

        eigenvalues, eigenvectors = np.linalg.eig(k_rot)
        if np.any(eigenvalues <= 0):
            print("Negative eigenvalues found in cell %d" % i)
            print("Eigenvalues: ", eigenvalues)
            break

        permeability_loc[i] = k_rot.flatten()

        if inverse:
            D[i, :] = np.linalg.inv(k_rot).flatten()
            return D

    return permeability_loc


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    print("All tests done.")
