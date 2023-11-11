import numpy as np

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
    k : list
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
    [2.0280701754385963e-11, 3.734728329018035e-12, 3.734728329018035e-12]
    >>> print(gebart(vf=0.5, rf=1.7e-5, packing="Quad", tensorial=True))
    [2.0280701754385963e-11, 0, 0, 0, 3.734728329018035e-12, 0, 0, 0, 3.734728329018035e-12]
    >>> print(gebart(vf=0.5, rf=1.7e-5, packing="Hex", tensorial=False))
    [2.181132075471698e-11, 4.727865993574315e-12, 4.727865993574315e-12]
    >>> print(gebart(vf=0.5, rf=1.7e-5, packing="Hex", tensorial=True))
    [2.181132075471698e-11, 0, 0, 0, 4.727865993574315e-12, 0, 0, 0, 4.727865993574315e-12]
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
        c1 =  16/9/np.pi/np.sqrt(2)
        vf_max = np.pi/4
        c = 57
    elif packing == "Hex":
        c1 =  16/9/np.pi/np.sqrt(6)
        vf_max = np.pi/2/np.sqrt(3)
        c = 53

    k1 = 8*rf**2/c * (1-vf)**3/vf**2
    k2 = c1 * rf**2 * np.sqrt( ( np.sqrt(vf_max/vf) - 1)**5 )

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

    return k


if __name__ == "__main__":
    import doctest
    doctest.testmod()