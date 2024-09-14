from .io import pk_save, pk_load, choose_directory, voxel2foam, voxel2inp
from .tow import Tow
from .mesh import background_mesh, label_mask, intersection_detect
from .geometry import Plane
from .misc import porosity_tow, gebart, cai_berdichevsky, drummond_tahir, perm_rotation
from .thirdparty.bcolors import bcolors
from .__dataset__ import example as pk_example

import zipfile
import os
import pyvista as pv
from tqdm.auto import tqdm
from scipy.sparse import coo_matrix
import numpy as np


class Textile:
    """
    A class representing a textile composed of multiple fiber tows. This class encapsulates the geometric and
    physical properties of a textile, providing functionalities for mesh generation, tow management,
    and export capabilities for simulation platforms such as OpenFOAM and Abaqus.

    Methods
    -------
    from_file(cls, path)
        Load a textile object from a file.
    __repr__()
        Representation of the textile.
    __getitem__(self, item)
        Get tow by tow name.
    add_tow(self, tow, group=None)
        Add a tow to the textile.
    add_group(self, name="group1", tow=None)
        Groups of the tows in the textile.
    remove(self, tow)
        Remove a tow from the textile.
    triangulate(self)
        Hexahedral mesh to tetrahedral mesh. Conformal meshing.
    decimate(self)
        Decimate the mesh.
    meshing(self, bbox, voxel_size=None, show=False, labeling=False, yarn_perm_model="Gebart",
            surface_mesh=None, verbose=False)
        Generate a mesh for the textile.
    cell_labeling(self, surface_mesh=None, intersection=False, check_surface=False, yarn_perm_model="Gebart",
                threshold=1, verbose=False)
        Label the cells of the background mesh with tow id.
    export_as_vtu(self, fp, binary=True)
        Export the textile mesh as a vtu file.
    export_as_openfoam(self, fp, scale=1, boundary_type=None, cell_data=["yarnIndex", "D"])
        Export the textile mesh as polyMesh folder for OpenFOAM simulation.
    export_as_inp(self, fp="./mesh-C3D8R.inp", scale=1, orientation=True)
        Export the textile mesh as inp file for Abaqus simulation.
    save(self, path=None, filename=None, data_size="minimal")
        Save the textile object to a file.
    reconstruct(self)
        Reconstruct the textile object from the saved file.
    case_prepare(self, path=None)
        Prepare a case for OpenFOAM simulation.
    """

    def __init__(self, name="textile"):
        """
        Initialize the textile object.
        """
        self.name = name
        self.__tows__ = {}
        self.groups = {}
        self.mesh = None
        self.mesh_shape = None
        self.mesh_bounds = None
        self.voxel_size = None
        self.__n_cells__ = None
        self.tex = []  # linear density of the tows in the textile

        self.data_size = "full"  # size of the data to be saved

    @classmethod
    def from_file(cls, path):
        """
        Load a textile object from a file.

        Parameters
        ----------
        path : str
            Path of the file to be loaded.

        Returns
        -------
        Textile object.
        """
        # check if the file is in the format of ".tex"
        if not path.endswith(".tex"):
            raise ValueError("The file must be in the format of '.tex'.")

        return pk_load(path)

    def __repr__(self):
        """
        Representation of the textile.
        """
        tow_str = str(self.__tows__.values()).replace(", ", ".\n - ")
        tow_str = tow_str.replace("Tow: ", "")
        return "Textile(name=%s, \ntows=%s)" % (self.name, tow_str)

    def __getitem__(self, item):
        """
        Get tow by tow name.
        """
        return self.__tows__[item]

    @property
    def n_tows(self):
        """
        Number of tows in the textile.
        """
        return len(self.__tows__)

    @property
    def items(self):
        """
        Return tow names as a list. The tow names are reordered by the tow number if the tow
        name is in the format of "towType_towNumber". Otherwise, the tow names are ordered
        according to the order of adding the tows to the textile.
        """
        # check if tow names are in the format of towType_towNumber

        tow_names = list(self.__tows__.keys())
        tow_numbers = []
        tow_names_new = []
        for tow_name in tow_names:
            # check if tow name is in the format of towType_towNumber: if not number, then return

            if "_" not in tow_name or (not tow_name.split("_")[1].isdigit()):
                print(bcolors.WARNING + "Warning: tow name %s is not in the format of towType_towNumber. "
                                        "The tow names are ordered \naccording to the order of adding the "
                                        "tows to the textile." % tow_name + bcolors.ENDC)
                return tow_names
            tow_number = tow_name.split("_")[1]
            tow_numbers.append(int(tow_number))
        tow_numbers = np.array(tow_numbers)
        tow_numbers_sort = np.argsort(tow_numbers)
        for i in tow_numbers_sort:
            tow_names_new.append(tow_names[i])

        return tow_names_new

    @property
    def bounds(self):
        """
        Bounding box of the textile.
        """
        # traverse all tows to get the bounds
        bbox = np.zeros([self.n_tows, 6])
        n = 0
        for tow in self.__tows__.values():
            bbox[n, :] = tow.bounds
            n += 1
        mask_min = [True, False, True, False, True, False]
        mask_max = [False, True, False, True, False, True]

        x_min, y_min, z_min = np.min(bbox, axis=0)[mask_min]
        x_max, y_max, z_max = np.max(bbox, axis=0)[mask_max]

        return np.array([x_min, x_max, y_min, y_max, z_min, z_max])

    def add_tow(self, tow, group=None):
        """
        Add a tow to the textile. If tow is already in the textile, then raise
        a ValueError.

            Parameters
            ----------
            tow : Tow object
                Tow to be added to the textile. Stored in self.__tows__ as a dictionary.
                Tow.name is the key, and tow is the value.
            group : str
                Group name of the tow. If group is None, then the tow is not added
                to any group. Stored in self.groups.

            Returns
            -------
            None.
        """
        if not isinstance(tow, Tow):
            raise TypeError("Input tow must be a Tow object.")

        if tow.name in self.__tows__.keys():
            raise ValueError("Tow already exists. Please use"
                             "another name for the new tow.")

        self.__tows__[tow.name] = tow

        if group is not None:
            self.add_group(name=group, tow=tow)

        self.tex.append(tow.tex)
        self.bounds

        return None

    def add_group(self, name="group1", tow=None):
        """
        Groups of the tows in the textile. Each is a group of Tow objects.
        if tow is None, then the group is empty.

            Parameters
            ----------
            name : str
                Name of the group.
            tow : Tow object
                Tow to be added to the group. If tow is None, then the group is empty.
                If tow is not None, then tow is added to the group. Besides, if tow is
                not in the self.__tows__ yet, it will be added to self.__tows__.

            Returns
            -------
            None.
        """
        if tow is None:
            if name in self.groups.keys():
                raise ValueError("Group already exists. Please use another name for the "
                                 "new group.")
            else:
                self.groups[name] = []
                return None

        if not isinstance(tow, Tow):
            raise TypeError("Input tow must be a Tow object.")

        if tow.name not in self.__tows__.keys():
            self.add_tow(tow)

        if name not in self.groups.keys():
            self.groups[name] = []
            self.groups[name].append(tow.name)
        elif tow.name not in self.groups[name]:
            self.groups[name].append(tow.name)
        else:
            raise ValueError("Tow already exists in the group.")

        return None

    def remove(self, tow):
        """
        Remove a tow from the textile.

            Parameters
            ----------
            tow : str or Tow object
                The tow to be removed.

            Returns
            -------
            None.
        """
        if isinstance(tow, Tow):
            tow = tow.name

        if tow not in self.__tows__.keys():
            raise ValueError("Tow does not exist.")

        self.__tows__.pop(tow)

        print("Tow %s is removed." % tow)

        # check if the tow is in any group
        if self.groups != {}:
            for group in self.groups.keys():
                if tow in groups[group]:
                    group.pop(tow)
                    print("Tow %s is removed from group %s." % (tow, group))

        return None

    def triangulate(self):
        """
        Hexahedral mesh to tetrahedral mesh. Conformal meshing.
        """

        pass

    def decimate(self):
        """
        Decimate the mesh.
        """
        pass

    def meshing(self, bbox, voxel_size=None, show=False, labeling=False, yarn_perm_model="Gebart",
                surface_mesh=None, verbose=False):
        """
        Generate a mesh for the textile.

            Parameters
            ----------
            bbox : numpy.ndarray
                bounding box of the background mesh specified through a numpy array
                contains the minimum and maximum coordinates of the bounding box
                [xmin, xmax, ymin, ymax, zmin, zmax]
            voxel_size : float, or numpy.ndarray
                voxel size of the background mesh.
                if `None`, the voxel size is set to the 1/20 of the diagonal length of the bounding box;
                if `float`, the voxel size is set to the float value in x, y, z directions;
                if `list`, `set`, or `tuple` of size 3, the voxel size is set to the values for
                the x-, y- and z- directions.
            show : bool, optional
                Whether to show the mesh. The default is False.
            labeling : bool, optional
                Whether to label the background mesh cells with tow id. The default is True.
            yarn_perm_model : str, optional
                The permeability model used to calculate the permeability tensor of the fiber tow.
                The default is "Gebart". The available permeability models are "Gebart", "CaiBerdichevsky",
                and "DrummondTahir".
            surface_mesh : str, optional
                Path of the surface meshes of fiber tows. The default is None. If `labeling=True`,
                then the surface meshes are loaded from the path specified by `surface_mesh`. The surface
                meshes are used to label the background mesh cells with tow id. If surface_mesh is `None`,
                then the surface meshes are selected by the user interactively.
                TODO : Use the surface mesh generated by the tow object.

            Returns
            -------
                None.
        """
        mesh_background, mesh_shape = background_mesh(bbox, voxel_size=voxel_size)
        self.mesh = mesh_background
        self.mesh_shape = mesh_shape
        self.mesh_bounds = bbox
        self.voxel_size = voxel_size
        self.__n_cells__ = mesh_background.n_cells

        cell_rending = None
        if labeling:
            self.cell_labeling(surface_mesh=surface_mesh, verbose=verbose, yarn_perm_model=yarn_perm_model)
            cell_rending = "yarnIndex"

        if show:
            self.mesh.plot(scalars=cell_rending, show_edges=True)

        return None

    def cell_labeling(self, surface_mesh=None, intersection=False, check_surface=False, yarn_perm_model="Gebart",
                      threshold=1, verbose=False):
        """
        Label the cells of the background mesh with tow id.

            Parameters
            ----------
            surface_mesh : pyvista mesh object (PolyData)
                Surface mesh of fiber tows. If `None`, then the surface meshes is selected
                user interactively. Otherwise, the surface mesh is loaded from the path
                specified by `surface_mesh`.
            intersection : bool, optional
                Whether to detect the intersection of the tows. The default is False.
            check_surface : bool, optional
                Whether to check if the surface mesh is watertight. The default is False.
            yarn_perm_model : str, optional
                The permeability model used to calculate the permeability tensor of the fiber tow.
                The default is "Gebart". The available permeability models are "Gebart", "CaiBerdichevsky",
                and "DrummondTahir".
            threshold : float, optional
                The tolerance for the fiber tow section detection. The default is 1. A wavy
                fiber tow may have several intersections with the plane of a cross-section. The
                threshold is used to determine if the cells is correctly labelled.
            verbose : bool, optional
                Whether to print the information. The default is False.

            Returns
            -------
                None.

            Notes
            -----
            Please make sure that the name of the tow (`tow.name`) is in the format of `towType_towNumber`.
        """

        """ Select the surface meshes of yarns to be labelled """
        if surface_mesh is None:
            path = choose_directory("Choose the surface mesh directory for fiber tow labelling")
        else:
            path = surface_mesh

        """ Initialize cell data arrays """
        label_list = np.full(self.__n_cells__, -1, dtype=np.int32)
        section_list = np.full(self.__n_cells__, -1, dtype=np.int32)
        area_list = np.full(self.__n_cells__, -1., dtype=np.float64)
        porosity_list = np.full(self.__n_cells__, 1., dtype=np.float64)
        permeability_list = np.full([self.__n_cells__, 9], 0., dtype=np.float64)
        permeability_local_list = np.full([self.__n_cells__, 9], 0., dtype=np.float64)
        resistance_list = np.full([self.__n_cells__, 9], 0., dtype=np.float64)
        orientation_list = np.full([self.__n_cells__, 3], 0., dtype=np.float64)

        label_set_dict = dict()  # dictionary of label sets for each yarn used for intersection detection

        print(bcolors.OKBLUE + "Labeling the background mesh cells with tow id ..." + bcolors.ENDC)

        for item in tqdm(self.items):
            tow = self[item]

            if verbose:
                print("Processing yarn: %s" % item)

            mesh_tri = pv.read(os.path.join(path, (item + ".stl")))  # load surface mesh

            # find the cells that are within the tubular surface of the fiber tow
            mask, label_yarn = label_mask(self.mesh, mesh_tri, tolerance=0.0000001, check_surface=check_surface)

            label_list[mask] = int(item.split("_")[1])
            label_set_dict[int(item.split("_")[1])] = coo_matrix(label_yarn)

            """ Label tow cross-sections """
            cell_centers = self.mesh.cell_centers().points
            yarn_centers = cell_centers[mask]

            orientation = tow.orientation
            centerline = tow.geom_features.iloc[:, -3:].to_numpy()
            temp_section = np.full([yarn_centers.shape[0], centerline.shape[0]], -1.)

            scale_factor = {"um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1}
            yarn_area = tow.geom_features.iloc[:, 0].to_numpy() * scale_factor[tow.length_scale] ** 2
            yarn_porosity = porosity_tow(tow.tex, yarn_area, rho_fiber=tow.rho_fiber)

            po_ = yarn_porosity.copy()
            if np.any(yarn_porosity < 0.15):
                yarn_porosity[yarn_porosity < 0.15] = np.average(yarn_porosity[yarn_porosity > 0.15])
            elif np.any(yarn_porosity > 1):
                raise ValueError("The porosity of the yarn is larger than 1.")

            if np.any(np.isnan(yarn_porosity)):
                yarn_porosity[np.isnan(yarn_porosity)] = 0.35
                print(
                    bcolors.WARNING + "Warning: The porosity of the yarn %s is nan. The porosity is set to 0.35." % item + bcolors.ENDC)

            if yarn_perm_model == "Gebart":
                yarn_permeability_local = np.array(
                    gebart(1 - yarn_porosity, rf=tow.radius_fiber, packing=tow.packing_fiber, tensorial=True))
            elif yarn_perm_model == "CaiBerdichevsky":
                yarn_permeability_local = np.array(
                    cai_berdichevsky(1 - yarn_porosity, rf=tow.radius_fiber, packing=tow.packing_fiber, tensorial=True))
            elif yarn_perm_model == "DrummondTahir":
                yarn_permeability_local = np.array(
                    drummond_tahir(1 - yarn_porosity, rf=tow.radius_fiber, packing=tow.packing_fiber, tensorial=True))

            # check if nan exists in the permeability tensor
            if np.any(np.isnan(yarn_permeability_local)):
                print("The permeability tensor of yarn %s contains nan." % item)
                raise ValueError("The permeability tensor of yarn %s contains nan." % item)

            perm_inv = True  # inverse the permeability tensor
            yarn_permeability, yarn_resistance = perm_rotation(yarn_permeability_local, orientation, inverse=perm_inv)

            for i in range(centerline.shape[0]):
                plane = Plane(p1=centerline[i, :], normal_vector=orientation[i, :])
                dist = plane.distance(yarn_centers)
                # calculate the distance of yarn centers to the centroid of the cross-section
                # TODO : A safer way to determine the threshold should be developed.
                centroid_dist = np.linalg.norm(yarn_centers - centerline[i, :], axis=1)
                mask_centroid = centroid_dist > threshold
                dist[mask_centroid] = 10e12
                temp_section[:, i] = dist

            idx = np.argmin(np.abs(temp_section), axis=1)

            section_list[mask] = idx
            area_list[mask] = yarn_area[idx]
            porosity_list[mask] = yarn_porosity[idx]
            permeability_list[mask, :] = yarn_permeability[idx, :]
            permeability_local_list[mask, :] = yarn_permeability_local[idx, :]
            resistance_list[mask, :] = yarn_resistance[idx, :]
            orientation_list[mask, :] = orientation[idx, :]

        self.mesh.cell_data['yarnIndex'] = label_list
        self.mesh.cell_data['yarnSection'] = section_list
        self.mesh.cell_data['area'] = area_list
        self.mesh.cell_data['porosity'] = porosity_list
        self.mesh.cell_data['K'] = permeability_list
        self.mesh.cell_data['K_local'] = permeability_local_list
        self.mesh.cell_data['D'] = resistance_list
        self.mesh.cell_data['orientation'] = orientation_list

        if intersection:
            """ Label the intersected cells"""
            label_list = intersection_detect(label_list, label_set_dict)

            intersect_info, intersect_info_dict, cell_data_intersect = intersection_detect(label_set_dict)
            self.mesh.cell_data['intersection'] = cell_data_intersect

        return None

    def export_as_vtu(self, fp, binary=True):
        """
        Export the textile mesh as a vtu file.

            Parameters
            ----------
            fp : str
                The file path of the output mesh.
            binary : bool, optional
                Whether to save the mesh in binary format. The default is True.

            Returns
            -------
            None.
        """
        if self.mesh is None:
            raise ValueError("The textile mesh is not generated yet. Please generate the textile "
                             "mesh with `Textile.meshing()` first.")

        self.mesh.save(fp, binary=binary)

        return None

    def export_as_openfoam(self, fp, scale=1, boundary_type=None, cell_data=["yarnIndex", "D"]):
        """
        Export the textile mesh as polyMesh folder for OpenFOAM simulation.

        The structure of the output case folder is:

        ::

             fp/ textile.name/
            ├── 0
            │   ├── D
            │   ├── yarnIndex
            │   └── ...
            └── constant
                  ├── polyMesh
                  │   ├── boundary
                  │   ├── faces
                  │   ├── neighbour
                  │   ├── owner
                  │   ├── points
                  │   ├── ...
                  └── ...

            Parameters
            ----------
            fp : str
                The file path of the output mesh.
            scale : float, optional
                The scale factor of the mesh. To convert the mesh from mm to m, the scale factor
                should be 0.001. The default is 1.
            boundary_type : dict, optional
                The boundary type of the mesh. The default is None. If None, the boundary type
                will be set as "wall" for all boundaries.
            cell_data : list, optional
                The cell data to be written into the mesh. The default is ["yarnIndex", "D"].

            Returns
            -------
            None.
        """
        if self.mesh is None:
            raise ValueError("The textile mesh is not generated yet. Please generate the textile "
                             "mesh with `Textile.meshing()` first.")

        if boundary_type is None:
            boundary_type = {"left": "wall", "right": "wall", "front": "wall", "back": "wall",
                             "bottom": "patch", "top": "patch"}

        suffix = "_".join([str(i) for i in self.mesh_shape])
        outputDirMesh = os.path.join(fp, self.name + "_" + suffix)

        n_copy = 0
        while os.path.exists(outputDirMesh):
            suffix = suffix + "_" + str(n_copy)
            outputDirMesh = os.path.join(fp, self.name + "_" + suffix)
            n_copy += 1

        print(bcolors.OKBLUE + "Exporting the textile mesh as OpenFOAM case to %s ..." % outputDirMesh + bcolors.ENDC)

        voxel2foam(self.mesh, scale=scale, outputDir=outputDirMesh, boundary_type=boundary_type,
                   cell_data_list=cell_data)

        # Create a markdown file in the output directory to record the textile information
        with open(os.path.join(outputDirMesh, "textile_info.md"), "w") as f:
            f.write("## Textile name \n {}\n".format(self.name))
            f.write("## Bounding box \n {}\n".format(self.mesh_bounds))
            f.write("## Voxel size \n {}\n".format(self.voxel_size))
            f.write("## Mesh shape \n {}\n".format(self.mesh_shape))
            f.write("## Number of yarns \n {}\n".format(len(self.items)))
            f.write("## Yarns \n {}\n".format(self.__repr__()))
            f.write("## Yarns linear density \n {}\n".format(self.tex))
        f.close()

        # Create a OpenFOAM case file for Paraveiw post-processing
        with open(os.path.join(outputDirMesh, self.name + "_" + suffix + ".OpenFOAM"), "w") as f:
            # empty file
            pass
        f.close()

        self.__case_root__ = outputDirMesh
        return None

    def export_as_inp(self, fp="./mesh-C3D8R.inp", scale=1, orientation=True):
        """
        Export the textile mesh as inp file for Abaqus simulation.

            Parameters
            ----------
            fp : str
                The file path and filename of the output mesh. The default is "./mesh-C3D8R.inp".
            scale : float, optional
                The scale factor of the mesh. To convert the mesh from mm to m, the scale factor
                should be 0.001. The default is 1.
            orientation : bool, optional
                Whether to export the orientation of the yarns. The default is True.

            Returns
            -------
            None.
        """
        if self.mesh is None:
            raise ValueError("The textile mesh is not generated yet. Please generate the textile "
                             "mesh with `Textile.meshing()` first.")

        mesh = self.mesh
        voxel2inp(mesh, scale=1, outputDir=fp, orientation=True)

    def save(self, path=None, filename=None, data_size="minimal"):
        """
        Save the textile object to a file.

            Parameters
            ----------
            path : str, optional
                Path of the file to be saved. The default is None. If `path` is None,
                then the user is asked to select the directory to save the file.
            filename : str, optional
                Filename of the file to be saved. The default is None. If `filename` is None,
                then the filename is set to the textile name with extension ".tex".
            data_size : str, optional
                Size of the data to be saved. The default is "minimal". If `data_size` is "minimal",
                then only the minimal information of the textile that can be used to reconstruct the
                textile object is saved. If `data_size` is "full", then all information of the textile
                is saved.

            Returns
            -------
            None.

            Notes
            -----
            TODO : more storage can be saved. The Textile and Tow classes should be designed carefully
                   at next version.
        """
        if path is None:
            path = choose_directory("Choose the directory to save the textile object.")

        if filename is None:
            filename = self.name + ".tex"
        elif not filename.endswith(".tex"):
            filename = filename + ".tex"

        path = os.path.join(path, filename)

        if data_size == "minimal":
            if self.mesh is not None:
                self.mesh = None
            for item in self.items:
                tow = self[item]
                for attr in ["_Tow__geom_features", "_Tow__coordinates", "_Tow__kriged_vertices",
                             "_Tow__surf_mesh", "_Tow__traj", "__orient__", "surf_points"]:
                    if hasattr(tow, attr):
                        delattr(tow, attr)
            self.data_size = "minimal"
        pk_save(path, self)

        return None

    def reconstruct(self):
        """
        Reconstruct the textile object from the saved file. The saved file must be
        in the format of ".tex" and loaded by the `pk_load` function.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        :noindex:
        """
        # meshing
        if self.data_size == "minimal":
            self.meshing(self.bounds, voxel_size=self.voxel_size, show=False, labeling=True,
                         surface_mesh=None, verbose=False)
        else:
            pass

        return None

    def case_prepare(self, path=None):
        """
        Prepare a case for OpenFOAM simulation.

        Parameters
        ----------
        path : str
            Path of the case to be prepared. The default is None. If `path` is None,
            it is set to the root directory of the OpenFOAM mesh generated by
            `Textile.export_as_openfoam()`.

        Returns
        -------
        None.

        :noindex:
        """
        if path is None:
            path = self.__case_root__

        pk_example("case template", outdir=path)
        with zipfile.ZipFile(path + "/CaseTemplate.zip", 'r') as zip_ref:
            filelist = zip_ref.namelist()
            for file in filelist:
                if file.split("/")[-1] in ["controlDict", "decomposeParDict",
                                           "flowRatePatch", "fvSchemes", "fvSolution"]:
                    sys_path = os.path.join(path, "system")
                    if not os.path.exists(sys_path):
                        os.makedirs(sys_path)
                    with open(os.path.join(sys_path, file.split("/")[-1]), 'wb') as f:
                        f.write(zip_ref.read(file))
                elif file.split("/")[-1] in ["Allrun", "Allclean", "PyFoamFileClean.py"]:
                    with open(os.path.join(path, file.split("/")[-1]), 'wb') as f:
                        f.write(zip_ref.read(file))
                elif file.split("/")[-1] in ["F", "p", "U"]:
                    ini_path = os.path.join(path, "0")
                    if not os.path.exists(ini_path):
                        os.makedirs(ini_path)
                    with open(os.path.join(ini_path, file.split("/")[-1]), 'wb') as f:
                        f.write(zip_ref.read(file))
                elif file.split("/")[-1] in ["momentumTransport", "transportProperties"]:
                    const_path = os.path.join(path, "constant")
                    if not os.path.exists(const_path):
                        os.makedirs(const_path)
                    with open(os.path.join(const_path, file.split("/")[-1]), 'wb') as f:
                        f.write(zip_ref.read(file))
        zip_ref.close()
        os.remove(path + "/CaseTemplate.zip")
        print(bcolors.OKBLUE + "Case preparation is done." + bcolors.ENDC)
        return None
