import os
import zipfile

import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R

from .thirdparty.bcolors import bcolors


def gebart(vf, rf, packing="Quad", tensorial=False):
    """
    Calculate the fiber tow permeability for a given fiber packing
    pattern according to Gebart's model.

    .. math::
            k_l & = \\frac{8 r_f^2}{c} \\frac{(1 - V_f)^3}{V_f^2}  \\\\
            k_t & = c_1 r_f^2 \\sqrt{\\left(\\sqrt{\\frac{V_{f_{max}}}{V_f}} - 1\\right)^5}

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

    # Calculate the maximum fiber volume fraction
    if packing == "Quad":
        vf_max = np.pi / 4
    elif packing == "Hex":
        vf_max = np.pi / 2 / np.sqrt(3)

    if not isinstance(vf, np.ndarray):
        vf = np.array(vf).flatten()

    k = np.zeros([vf.shape[0], 9])

    if np.any(vf < 0) or np.any(vf >= vf_max):
        raise ValueError(f"Fiber volume fraction must be between 0 and {vf_max}.")

    mask = ~(vf == 0)  # mask for non-zero fiber volume fraction
    vf = vf[mask]

    rf = np.array(rf)
    if np.any(rf <= 0):
        raise ValueError("Fiber radius must be positive.")

    # Calculate permeability
    if packing == "Quad":
        c1 = 16 / 9 / np.pi / np.sqrt(2)
        c = 57
    elif packing == "Hex":
        c1 = 16 / 9 / np.pi / np.sqrt(6)
        c = 53

    k_l = 8 * rf ** 2 / c * (1 - vf) ** 3 / vf ** 2
    k_t = c1 * rf ** 2 * np.sqrt((np.sqrt(vf_max / vf) - 1) ** 5)

    k[mask, 0] = k_l
    k[mask, 4] = k_t
    k[mask, 8] = k_t

    if tensorial:
        return k
    else:
        return k[:, [0, 4, 8]]


def cai_berdichevsky(vf, rf, packing='Quad', tensorial=False):
    """
    Calculate the fiber tow permeability for a given fiber packing
    pattern according to cai's model.

    .. math::
        K_L = & 0.211 r^2\\left(\\left(V_a-0.605\\right)\\left(\\frac{0.907 V_f}{V_a}\\right)^{(-0.181)} *\\left(\\frac{1-0.907 V_f}{V_a}\\right)^{(2.66)}\\right. \\\\
        & \\left.+0.292\\left(0.907-V_a\\right)\\left(V_f\\right)^{(-1.57)}\\left(1-V_f\\right)^{(1.55)}\\right)   \\\\
        K_T = & 0.229 r^2\\left(\\frac{1.814}{V_a}-1\\right)\\left(\\frac{\\left(1-\\sqrt{\\frac{V_f}{V_a}}\\right)}{\\sqrt{\\frac{V_f}{V_a}}}\\right)^{2.5}

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
    Cai, Z. and A. Berdichevsky, An improved self‐consistent method for estimating the permeability of a fiber assembly. Polymer composites, 1993. 14(4): p. 314-323

    Examples
    --------
    >>> rf = 6.5e-6
    >>> k = cai_berdichevsky(vf=0.3, rf=rf, packing='Hex')
    >>> k / rf**2
    array([[0.04415829, 0.10741589, 0.10741589]])
    >>> k = cai_berdichevsky(vf=0.7, rf=rf, packing='Hex')
    >>> k / rf**2
    array([[0.00604229, 0.00162723, 0.00162723]])
    >>> k = cai_berdichevsky(vf=0.3, rf=rf, packing='Hex', tensorial=True)
    >>> k / rf**2
    array([[0.04415829, 0.        , 0.        , 0.        , 0.10741589,
            0.        , 0.        , 0.        , 0.10741589]])
    """

    # Check input
    if packing not in ['Quad', 'Hex']:
        raise ValueError('Invalid packing.Valid options are "Quad"and "Hex".')

    # Calculate the maximum fiber volume fraction
    if packing == "Quad":
        vf_max = np.pi / 4  # 0.7854
    elif packing == "Hex":
        vf_max = np.pi / 2 / np.sqrt(3)  # 0.9069

    if not isinstance(vf, np.ndarray):
        vf = np.array(vf).flatten()

    k = np.zeros([vf.shape[0], 9])

    if np.any(vf <= 0) or np.any(vf >= vf_max):
        raise ValueError(f"Fiber volume fraction must be between 0 and {vf_max}.")

    mask = ~(vf == 0)  # mask for non-zero fiber volume fraction
    vf = vf[mask]

    rf = np.array(rf)
    if np.any(rf <= 0):
        raise ValueError("Fiber radius must be positive.")

    # longtudinal permeability
    k_l = 0.211 * rf ** 2 * ((vf_max - 0.605) * (0.907 * vf / vf_max) ** (-0.181) * \
                             ((1 - 0.907 * vf) / vf_max) ** 2.66 + \
                             0.292 * (0.907 - vf_max) * vf ** (-1.57) * (1 - vf) ** 1.55)

    # transverse permeability
    k_t = 0.229 * rf ** 2 * (1.814 / vf_max - 1) * \
          np.sqrt((1 - np.sqrt(vf / vf_max)) / np.sqrt(vf / vf_max)) ** 5

    k[mask, 0] = k_l
    k[mask, 4] = k_t
    k[mask, 8] = k_t

    if tensorial:
        return k
    else:
        return k[:, [0, 4, 8]]


def drummond_tahir(vf, rf, packing='Quad', tensorial=False):
    """
    Calculate the fiber tow permeability for a given fiber packing
    pattern according to Drummond and Tahir's model.

    .. math::
        K_l & = \\frac{r^2}{4V_f}\\left(-lnV_f-1.476+2V_f-0.5V_f^2\\right)  \\\\
        K_{tQuad} & =\\frac{r^2}{8V_f}\\left(-lnV_f-1.476+\\frac{2V_f-0.796V_f}{1+0.489V_f-1.605{V_f}^2}\\right)  \\\\
        K_{tHex} & =\\frac{r^2}{8V_f}\\left(-lnV_f-1.497+2V_f-\\frac{V_f^2}{2}-0.739V_f^4+\\frac{2.534V_f^5}{1+1.2758V_f}\\right)

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
    Drummond J E, Tahir M I. Laminar viscous flow through regular arrays of parallel solid cylinders[J]. International Journal of Multiphase Flow, 1984, 10(5): 515-540..

    Examples
    --------
    >>> rf = 6.5e-6
    >>> k = drummond_tahir(vf=0.3, rf=rf, packing='Hex')
    >>> k / rf**2
    array([[0.23581067, 0.10851671, 0.10851671]])
    >>> k = drummond_tahir(vf=0.7, rf=rf, packing='Hex')
    >>> k / rf**2
    array([[0.01274105, 0.01110984, 0.01110984]])
    >>> k = drummond_tahir(vf=0.3, rf=rf, packing='Hex', tensorial=True)
    >>> k / rf**2
    array([[0.23581067, 0.        , 0.        , 0.        , 0.10851671,
            0.        , 0.        , 0.        , 0.10851671]])
    """
    # Check input
    if packing not in ['Quad', 'Hex']:
        raise ValueError('Invalid packing.Valid options are "Quad"and "Hex".')

    # Calculate the maximum fiber volume fraction
    if packing == "Quad":
        vf_max = np.pi / 4  # 0.7854
    elif packing == "Hex":
        vf_max = np.pi / 2 / np.sqrt(3)  # 0.9069

    if not isinstance(vf, np.ndarray):
        vf = np.array(vf).flatten()

    k = np.zeros([vf.shape[0], 9])

    if np.any(vf < 0) or np.any(vf >= vf_max):
        raise ValueError(f"Fiber volume fraction must be between 0 and {vf_max}.")

    mask = ~(vf == 0)  # mask for non-zero fiber volume fraction
    vf = vf[mask]

    rf = np.array(rf)
    if np.any(rf <= 0):
        raise ValueError("Fiber radius must be positive.")

    # longtudinal permeability
    k_l = (rf ** 2 / (4 * vf)) * (-np.log(vf) - 1.476 + 2 * vf - 0.5 * vf ** 2)

    # transverse permeability
    if packing == "Quad":
        k_t = (rf ** 2 / (8 * vf)) * (-np.log(vf) - 1.476 + (2 * vf - 0.796 * vf) / (1 + 0.489 * vf - 1.605 * vf ** 2))
    elif packing == "Hex":
        k_t = (rf ** 2 / (8 * vf)) * (
                    -np.log(vf) - 1.497 + 2 * vf - vf ** 2 / 2 - 0.739 * vf ** 4 + 2.534 * vf ** 5 / (1 + 1.2758 * vf))

    k[mask, 0] = k_l
    k[mask, 4] = k_t
    k[mask, 8] = k_t

    if tensorial:
        return k
    else:
        return k[:, [0, 4, 8]]


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


def perm_rotation(permeability, orientation, inverse=False, disable_tqdm=True):
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
        D : ndarray
            The inverse of the rotated permeability tensor. Shape: (n, 9)
    """
    if not isinstance(permeability, np.ndarray):
        permeability = np.array(permeability)
        assert permeability.shape[1] == 9, "The shape of permeability should be (n, 9)."

    if not isinstance(orientation, np.ndarray):
        orientation = np.array(orientation)
        assert orientation.shape[1] == 3, "The shape of orientation should be (n, 3)."

    D = np.zeros_like(permeability)
    permeability_loc = np.zeros_like(permeability)

    for i in tqdm(range(permeability.shape[0]), disable=disable_tqdm):

        if np.all(permeability[i] == 0):
            continue

        perm = np.reshape(permeability[i], (3, 3))
        ori = orientation[i]
        r = R.align_vectors([ori], [[1, 0, 0]])[0]
        A = r.as_matrix()
        k_rot = np.dot(np.dot(A, perm), A.T)

        eigenvalues, eigenvectors = np.linalg.eig(k_rot)
        if np.any(eigenvalues < 0):
            print("Negative eigenvalues found in cell %d" % i)
            print("Eigenvalues: ", eigenvalues)
            break
        elif np.any(eigenvalues == 0):
            print("Zero eigenvalues found in cell %d" % i)
            print("Eigenvalues: ", eigenvalues)
            break

        permeability_loc[i] = k_rot.flatten()

        if inverse:
            D[i, :] = np.linalg.inv(k_rot).flatten()

    return permeability_loc, D if inverse else permeability_loc


def compress_file(zipfilename, dirname):
    """
    Compresses all files and subdirectories in the specified directory.

    Parameters
    ----------
    zipfilename : str
        The name of the zip file, including the path.
    dirname : str
        The name of the directory to be compressed, including the path.

    Returns
    -------
    int
        1 if the compression is successful, otherwise 0.

    Examples
    --------
    >>> compress_file("test.zip", "./test")
    """
    # Ensure the directory exists
    if not os.path.exists(dirname):
        print(f"Directory '{dirname}' does not exist.")
        return

    # Ensure zipfilename is not a subdirectory of dirname to avoid recursive compression
    if zipfilename.startswith(dirname + os.path.sep):
        print("Error: The zip file name cannot be a subdirectory of the specified directory.")
        return

    try:
        # Create a ZipFile object for writing the compressed file
        with zipfile.ZipFile(zipfilename, 'w') as z:
            # Traverse all files and subdirectories in the specified directory
            for root, dirs, files in os.walk(dirname):
                for single_file in files:
                    # Construct the full path of the file
                    filepath = os.path.join(root, single_file)

                    # Add the file to the zip, using a relative path
                    z.write(filepath, os.path.relpath(filepath, dirname))
    except Exception as e:
        print(f"Error occurred while compressing files: {e}")
        return 0

    return 1


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    print("All tests done.")
