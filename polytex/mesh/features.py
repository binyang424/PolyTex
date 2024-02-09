import numpy as np
import pyvista as pv


def voxelize(mesh, density=None, check_surface=True, density_type='cell_number', contained_cells=False):
    """
    Voxelize surface mesh to UnstructuredGrid. The bounding box of the voxelized mesh possibly smaller
    than the bounding box of the surface mesh when cell_size type of density is used.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Surface mesh to be voxelized.
    density : float, int, or list of float or int
        Uniform size of the voxels when single float passed. A list of densities along
        x,y,z directions. Defaults to 1/100th of the mesh length for cell_size (float or list)
        flavor density and 50 cells in each direction for cell_number density (int or list).
    check_surface : bool
        Specify whether to check the surface for closure. If on, then the algorithm
        first checks to see if the surface is closed and manifold. If the surface is
        not closed and manifold, a runtime error is raised.
    density_type : str
        Specify the type of density to use. Options are 'cell_number' or 'cell_size'.
        When 'cell_number' is used, the density is the number of cells in each direction.
        When 'cell_size' is used, the density is the size of cells in each direction.
    contained_cells : bool
        If True, only cells that fully are contained in the surface mesh will be selected.
        If False, extract the cells that contain at least one of the extracted points.

    Returns
    -------
    vox : pyvista.UnstructuredGrid
        Voxelized unstructured grid of the original mesh.
    ugrid : pyvista.UnstructuredGrid
        The backgrond mesh for voxelization

    Examples:
    Create an equal density voxelized mesh using cell_size density.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> import polykriging.mesh as ms
    >>> mesh = pv.PolyData(examples.load_uniform().points)
    >>> vox, _ = ms.voxelize(mesh, density=0.5, density_type='cell_size')
    >>> vox.plot(show_edges = True)

    Create a voxelized mesh with specified number of elements in x, y, and z dimensions.

    >>> mesh = pv.PolyData(examples.load_uniform().points)
    >>> vox, _ = ms.voxelize(mesh, density=[50, 50, 50], density_type='cell_number', contained_cells=False)
    >>> vox.plot(show_edges = True)

    """
    import numpy as np
    import pyvista

    if not pyvista.is_pyvista_dataset(mesh):
        mesh = pyvista.wrap(mesh)
    if density is None:
        density = mesh.length / 100
    if isinstance(density, (int, float)):
        density_x, density_y, density_z = [density] * 3
    if isinstance(density, (list, set, tuple)):
        density_x, density_y, density_z = density

    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    if density_type == 'cell_number':
        density_x = int(density_x) + 1
        density_y = int(density_y) + 1
        density_z = int(density_z) + 1
        x = np.linspace(x_min, x_max, density_x, endpoint=True)
        y = np.linspace(y_min, y_max, density_y, endpoint=True)
        z = np.linspace(z_min, z_max, density_z, endpoint=True)
    elif density_type == 'cell_size':
        x = np.arange(x_min, x_max, density_x)
        y = np.arange(y_min, y_max, density_y)
        z = np.arange(z_min, z_max, density_z)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pyvista.StructuredGrid(x, y, z)
    ugrid = pyvista.UnstructuredGrid(grid)

    # get part of the mesh within the mesh's bounding surface.
    selection = ugrid.select_enclosed_points(
        mesh.extract_surface(), tolerance=0.0, check_surface=check_surface
    )
    mask = selection.point_data['SelectedPoints'].view(np.bool_)

    # extract cells from point indices
    adjacent_cells = not contained_cells
    vox = ugrid.extract_points(mask,adjacent_cells=adjacent_cells)
    return vox, ugrid


if __name__ == '__main__':
    # voxelize the mesh
    mesh = pv.read("./00_surfaceMesh/profile_0.stl")

    # test the effect of density on mesh volume and cell size
    density = [0.2, 1, 0.1]
    vox, _ = voxelize(mesh, density=density, density_type='cell_size', contained_cells=False)

    vox.plot(edge_color='k', show_edges=True)
