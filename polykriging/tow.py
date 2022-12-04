from .fileio import pk_save, pk_load, save_ply
from .fileio import save as save_tow

from .geometry import geom_tow, Curve, Polygon, Tube, ParamCurve
from .kriging import curve2Dinter, surface3Dinterp
from .mesh import get_cells
from .stats import kdeScreen, bw_scott
import numpy as np
import os
import pyvista as pv
# pv.set_plot_theme("document")


class Tow:
    """
    This class is used to store the tow information, calculate the geometry features
    and parametrilize the tow.

    Attributes
    ----------
    geom_features : dataframe
        The geometry features of each cross-section of the tow. The features include
        the 'Area', 'Perimeter', 'Width', 'Height', 'AngleRotated', 'Circularity',
       'centroidX', 'centroidY', and 'centroidZ'.
    coordinates : dataframe
        The parametrilized coordinates of the tow. It includes the geodesic 'distance'
        and the normalized distance 'norm_distance' from the start point of the parametric
        plane to the current point, the 'angular position (degree)', and the corresponding
        'X', 'Y', and 'Z' coordinates.

    Examples
    --------
    >>> import polykriging as pk
    >>> import numpy as np
    >>>
    >>> surf_points = pk.pk_load("./216_302_binder_1.pcd").to_numpy()
    >>> coordinates = surf_points[:, [2, 1, 0]] * 0.022  # convert voxel to mm
    >>> tow = pk.Tow(surf_points=coordinates, tex=0, name="Tow") # PolyKriging Tow class
    >>> df_coord = tow.coordinates  # parametric coordinates of the tow
    >>> df_geom = tow.geom_features  # geometrical features of the tow
    >>>
    >>> sample_position = np.linspace(0, 1, 20, endpoint=True)
    >>> pts_krig, expr_krig = tow.krig_cs(krig_config=("lin", "cub"),
                                          skip=10, sample_position=sample_position,
                                          smooth=0.0001)
    >>> mesh = tow.surf_mesh(order="zyx", plot=True)
    >>> mesh.save("./216_302_binder_1.vtk")
    >>> trajectory_sm = tow.trajectory(smooth=0.0015, plot=False, save_path="./trajectory.ply", orientation=True)
    """

    def __init__(self, surf_points, order,
                 tex=0, name="Tow", sort=True, resolution=None,
                 **kwargs):
        """

        Parameters
        ----------
        surf_points : str, ndarray or DataFrame
            The surface points of the tow should be an array of shape (n, 3)
            where n is the number of points. The points are extracted from
            volumetric images slice by slice. Please always put the column
            that indicates the slice number as the last column. It serves
            as the label for differentiate the points from different slices.

            If surf_points is a string, it should be the path to the file
            that stores the surface points. The file should be a .pcd file
            as defined in the PolyKriging library.

            If surf_points is a numpy array, it should be an array of shape
            (n, 3) where n is the number of points.

            If surf_points is a pandas DataFrame, it should be a DataFrame
            with 3 columns and n rows where n is the number of points.
        order : str
            It is preferred to set the last column of the surf_points as the coordinate in
            the direction that is perpendicular to the image slices for geometry analysis,
            parametrization and kriging resampling. Hence, you may have reordered the columns.
            Here, you can specify the order of the columns in the reordered points. Default is "xyz".
            The other options are "xzy", "yxz", "yzx", "zxy", "zyx". This function will recover the
            original order of the columns when generating the surface mesh.
        tex : float, optional
            The linear density of the tow in tex. Default is 0.
        name : str, optional
            The name or type of the tow. Default is "Tow".
        """
        self.name = name
        self.tex = tex
        self.order = order
        self.resolution = resolution

        if isinstance(surf_points, str):
            self.surf_points = pk_load(surf_points)
        elif isinstance(surf_points, np.ndarray):
            self.surf_points = surf_points
        # check if it is dataframe
        elif hasattr(surf_points, "to_numpy"):
            self.surf_points = surf_points.to_numpy()
        else:
            try:
                self.surf_points = np.array(surf_points, dtype=np.float32)
            except ValueError:
                raise ValueError("surf_points must be a point cloud data file (.pcd),"
                                 "a numpy array, or an pandas dataframe with xyz coordinates"
                                 "in the shape of (n,3).")
        self.geom_features, self.coordinates = geom_tow(self.surf_points, sort=sort)

    def __str__(self):
        return self.name

    def __repr__(self):
        tow_info = "Tow: " + self.name + "; \n" + \
                   "Linear density: " + str(self.tex) + "; \n" + \
                   "Number of points: " + str(self.surf_points.shape[0]) + "; \n" + \
                   "Number of cross-sections: " + str(self.geom_features.shape[0])
        return tow_info

    @property
    def __column_order__(self):
        # convert the letters in self.order to lower case
        order = self.order.lower()

        map = {"x": 0, "y": 1, "z": 2}
        order_list = [map[i] for i in order]
        # sort order and get the inverse permutation
        inv_order = np.argsort(order_list)
        return inv_order

    def krig_cs(self, krig_config=("lin", "cub"), skip=5, sample_position=[], smooth=0):
        """
        Kriging the cross-sections of the tow to oobtain the parametric equations
        for each cross-section (which is a closed curve).

        Parameters
        ----------
        krig_config : tuple, optional
            The kriging configuration. The first element is the kriging model for
            the drift term and the second is the name of covariance function. Default
            is ("lin", "cub").
        skip : int, optional
            The number of cross-sections to skip when kriging to accelerate the
            interpolation. Default is 5. If skip is 1, all the cross-sections will
            be kriged.
        sample_position : array_like, optional
            The resampling positions of each cross-sections specified by the normalized
            distance in radial direction of each cross-section. Default is [].
        smooth : float, optional
            The smoothing parameter for the kriging resampling. Default is 0. Also known
            as the nugget effect in geo-statistics and kriging theory.

        Returns
        -------
        _Tow__kriged_vertices : ndarray
            The kriged points for eacg cross-section of the tow. It is an array of shape
            (n, 3) where n is the number of points. The kriged points is obtained according
            to the kriging configuration and the resampling positions. If the resampling
            positions are not specified, the kriged points are obtained by evenly sampling
            the cross-sections with a sampling interval of 0.05, namely, 20 points per cross-
            section.
        expr : dict
            The kriging expression for each cross-section. It contains two sub-dictionaries that
             use the cross-section number as the key and the kriging expression as the value for
             the components in y and z directions. (x is the normalized distance in radial direction).
        """
        slices = np.unique(self.coordinates["Z"])[::skip]
        n_slices = len(slices)
        dict_cs_x = {}
        dict_cs_y = {}
        if len(sample_position) == 0:
            interp = np.linspace(0, 1, 20, endpoint=True)
        else:
            interp = sample_position

        self.theta_res = len(interp)
        self.h_res = n_slices

        pts_krig = np.zeros((n_slices * len(interp), 3))
        print("Kriging cross-sections... \n"
              "It may take a while depending on the number of cross-sections.")

        for i in range(n_slices):
            drift, cov = krig_config
            try:
                mask = self.coordinates["Z"] == slices[i]
                pts_cs = self.coordinates[mask]

                x_inter, x_expr = curve2Dinter(pts_cs.iloc[:, [1, 3]].to_numpy(),
                                               drift, cov, nuggetEffect=smooth, interp=interp)
                y_inter, y_expr = curve2Dinter(pts_cs.iloc[:, [1, 4]].to_numpy(),
                                               drift, cov, nuggetEffect=smooth, interp=interp)
                z_ = np.full(len(interp), slices[i])

                dict_cs_x[slices[i]] = x_expr
                dict_cs_y[slices[i]] = y_expr

                pts_krig[i * len(interp): (i + 1) * len(interp), :] = np.vstack((x_inter, y_inter, z_)).T

                if i % 10 == 0:
                    print("Kriging cross-sections... {}%".format(round(i / n_slices * 100, 2)))
                    print(i, "th cross-section", pts_cs.shape)

            except np.linalg.LinAlgError:
                print("Kriging failed at slice: {} because of singular matrix error. This could be avoided "
                      " by using a non-zero smooth value.".format(slices[i]))
                continue

        expr = {self.order[0] + " kriging equation": dict_cs_x, self.order[1] + " kriging equation": dict_cs_y}
        print("Kriging on cross-sections is finished.")

        self.__kriged_vertices = pts_krig  # the columns are in the same order as the input data (surf_points)
        inv_order = self.__column_order__
        return pts_krig[:, inv_order], expr

    def surf_mesh(self, plot=False, save_path=None, end_closed=False):
        """
        Generate the surface mesh of the tow.

        Parameters
        ----------
        plot : bool, optional
            Whether to plot the surface mesh. Default is False.
        save_path : str, optional
            The path to save the surface mesh. Default is None and the surface mesh will not be saved.
            The file format can be .ply or .vtk.
        Returns
        -------
        surf_mesh :
            The surface mesh of the tow.
        """
        # if self._Tow__kriged_vertices is not defined, raise error
        if not hasattr(self, "_Tow__kriged_vertices"):
            raise AttributeError("The kriged cross-sections are not defined. Please generate the "
                                 "kriged cross-section points using Tow.krig_cs() first.")

        inv_order = self.__column_order__

        h_res = int(np.unique(self._Tow__kriged_vertices[:, 2]).shape[0])
        theta_res = int(self._Tow__kriged_vertices.shape[0] / h_res)

        pts = self._Tow__kriged_vertices[:, inv_order]

        tube = Tube(theta_res, h_res, vertices=pts)
        mesh = tube.mesh(plot=plot)

        if save_path is not None:
            mesh = tube.save_as_mesh(save_path, end_closed=end_closed)
        return mesh

    def trajectory(self, krig_config=("lin", "cub"), smooth=0.0, plot=False, save_path=None, orientation=False):
        """
        Generate the trajectory of the tow and smooth it using kriging.

        Parameters
        ----------
        krig_config : tuple, optional
            The kriging configuration for the trajectory. It is a tuple of two strings that
            specify the drift and covariance function. Default is ("lin", "cub").
        smooth : float, optional
            The smoothing parameter for the parametric curve kriging. Default is 0. Also known
            as the nugget effect in geo-statistics and kriging theory.
        plot : bool, optional
            Whether to plot the trajectory. Default is False.
        save_path : str, optional
            The path to save the trajectory as vtk file. Default is None and the trajectory
            will not be saved.
        orientation : bool, optional
            Whether to calculate the orientation of the tow. Default is False. The orientation
            is the tangent vector of the trajectory.

        Returns
        -------
        traj : np.ndarray
            The trajectory of the tow in the form of (n, 3) where n is the number of points.
        """
        dataset = self.geom_features.iloc[:, [-3, -2, -1]].to_numpy()
        dataset = dataset[:, [-1, -3, -2]]

        # normalize the first column of dataset
        # hard copy dataset[:, 0] to avoid changing the original dataset
        dataset_0_org = dataset[:, 0].copy()

        dataset[:, 0] = dataset[:, 0] / dataset[:, 0].max()

        if smooth != 0.0:

            centerline = ParamCurve(("s", 0, 1), dataset=dataset, krig_config=krig_config, smooth=smooth)
            cen = centerline.eval(dataset[:, 0])

            traj = np.hstack((dataset_0_org.reshape(-1, 1),
                              cen.points))[:, [1, 2, 0]][:, self.__column_order__]
            cen_points = cen.points
        else:
            cen_points = dataset[:, 1:]
            traj = np.hstack((dataset_0_org.reshape(-1, 1),
                              cen_points))[:, [1, 2, 0]][:, self.__column_order__]

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(dataset_0_org, cen_points[:, 0], "r")
            plt.plot(dataset_0_org, cen_points[:, 1], "b")
            plt.gca().set_aspect("equal")

        self.__traj = traj

        if orientation:
            # calculate the orientation of the tow
            # the orientation is defined as the tangent vector of the centerline.

            orient = np.zeros((dataset.shape[0], 3))

            x_dist = np.diff(dataset_0_org) + 1e-15

            # TODO: replace with central difference for non-homogeneous stepsize as described in:
            # https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
            orient[1:-1, 0] = [(cen_points[i + 1, 0] - cen_points[i - 1, 0]) / (x_dist[i - 1] + x_dist[i])
                               for i in range(1, cen_points.shape[0] - 1)]
            orient[1:-1, 1] = [(cen_points[i + 1, 1] - cen_points[i - 1, 1]) / (x_dist[i - 1] + x_dist[i])
                               for i in range(1, cen_points.shape[0] - 1)]

            orient[:, 2] = 1

            # TODO : backward and forward difference for the first and last point
            orient[0, :] = orient[1, :]
            orient[-1, :] = orient[-2, :]

            # normalize the tangent vector
            orient = self.unit_vector(orient, ax=1)

            orient = orient[:, self.__column_order__]

            self.__orient__ = orient
            self.orientation = orient

        if save_path is not None:
            import meshio
            cells = [("line", [[x, x + 1] for x in np.arange(traj.shape[0]).tolist()[:-1]])]
            mesh = meshio.Mesh(points=traj,
                               cells=cells,
                               point_data={"nx": orient[:, 0],
                                           "ny": orient[:, 1],
                                           "nz": orient[:, 2]},
                               )
            try:
                mesh.write(save_path)
                print("The trajectory is saved.")
            except meshio._exceptions.ReadError:
                print("The file format is not supported or the file format"
                      "can not be deduced from path {}. Please use \".ply.\"".format(save_path))

        return traj

    def unit_vector(self, vector, ax=1):
        """
        Returns the unit vector of the input vector along the given axis.

        Parameters
        ----------
        vector : ndarray
            The input vector.
        ax : int, optional
            The axis along which the unit vector is calculated. Default is 1 (column).

        Returns
        -------
        unit_vector : ndarray
            The unit vector of the input vector along the given axis.
        """
        norm = np.zeros(vector.shape)
        norm_value = np.linalg.norm(vector, axis=ax)
        for i in np.arange(vector.shape[1]):
            norm[:, i] = norm_value
        return vector / norm

    def smoothing(self,
                  name_drift=['lin', 'lin'],
                  name_cov=['cub', 'cub'],
                  smooth_factor=[0, 0],
                  size=25, save_path=None, plot=False):
        """
        Smooth the tow using parametric surface kriging.
        Anisotropic smoothing is applied to the tow by using different smoothing factors
        for the two directions.

        Parameters
        ----------
        name_drift : list, optional
            The drift function for the parametric surface kriging. Default is ['lin', 'lin'].
        name_cov : list, optional
            The covariance function for the parametric surface kriging. Default is ['cub', 'cub'].
        smooth_factor : list, optional
            The smoothing factor for the parametric surface kriging. Default is [0, 0]. Also known
            as the nugget effect in geo-statistics and kriging theory. The smoothing factor
            is applied to the two directions separately. The first element is the smoothing
            factor for the radial direction and the second element is the smoothing factor for
            the axial direction.
        size : int, optional
            # TODO : improve the description
            The size of the window for the moving average filter. Default is 25.
        save_path : str, optional
            The path to save the smoothed tow. Default is None. If None, the smoothed tow
            will not be saved. The file format is .ply.
        plot : bool, optional
            Whether to plot the smoothed tow. Default is False.

        Returns
        -------
        vertices : ndarray
            The smoothed tow vertices.
        """
        import time
        start = time.time()

        labels = [self.order[0], self.order[1], self.order[2]]

        col_0 = self._Tow__kriged_vertices[:, 0]
        col_1 = self._Tow__kriged_vertices[:, 1]
        col_2 = self._Tow__kriged_vertices[:, 2]

        s = self.clusters['cluster centers']
        t = np.unique(col_2) / np.max(col_2)  # the input order

        theta_res = s.size
        h_res = t.size

        wins_interp, wins_result = self.smooth_window(size, h_res)

        # print("wins_interp: ", wins_interp)
        # print("wins_result: ", wins_result)

        x, y, z = col_0.reshape(-1, theta_res), \
                  col_1.reshape(-1, theta_res), \
                  col_2.reshape(-1, theta_res)

        x, y, z = x.T, y.T, z.T

        x_interp = np.zeros((theta_res, h_res))
        y_interp = np.zeros((theta_res, h_res))
        z_interp = np.zeros((theta_res, h_res))

        element_idx = 0
        for i in np.arange(wins_interp.shape[0]):
            print("{} iteration(s):".format(i))

            win_interp = wins_interp[i, :]
            win_result = wins_result[i, :]

            eff_elements = win_result[1] - win_result[0]

            print("    First component ... ")
            x_temp = surface3Dinterp(s, t[win_interp[0]:win_interp[1]],
                                     x[:, win_interp[0]:win_interp[1]],
                                     name_drift, name_cov,
                                     smooth_factor, label=labels[0])

            print("    Second component ... ")
            y_temp = surface3Dinterp(s, t[win_interp[0]:win_interp[1]],
                                     y[:, win_interp[0]:win_interp[1]],
                                     name_drift, name_cov,
                                     smooth_factor, label=labels[1])

            # print("    Third component ... ")
            # z_temp = surface3Dinterp(s, t[win_interp[0]:win_interp[1]],
            #                          z[:, win_interp[0]:win_interp[1]],
            #                          name_drift, name_cov,
            #                          [1e-8, 1e-8], label=labels[2])

            x_interp[:, element_idx:element_idx + eff_elements] = x_temp[:, win_result[0]:win_result[1]]
            y_interp[:, element_idx:element_idx + eff_elements] = y_temp[:, win_result[0]:win_result[1]]
            # z_interp[:, element_idx:element_idx + eff_elements] = z_temp[:, win_result[0]:win_result[1]]

            element_idx += eff_elements

        # x_interp[0, :] = x_interp[-1, :]
        # y_interp[0, :] = y_interp[-1, :]
        # z_interp[0, :] = z_interp[-1, :]

        print("Smoothing done! It takes {} seconds.".format(time.time() - start))

        z_interp = z
        vertices = np.vstack((x_interp.flatten("F"),
                              y_interp.flatten("F"),
                              z_interp.flatten("F"))).T

        inv_order = self.__column_order__
        vertices = vertices[:, inv_order]

        self.smoothed_vertices = vertices

        tube = Tube(theta_res, h_res, vertices=vertices)
        if save_path is not None:
            tube.save_as_mesh(save_path)
        if plot:
            tube.mesh(plot=True)
        return vertices

    def kde(self, bw=None, save_path=None):
        """
        Generate the kernel density estimation of the radial normalized distance for point cloud
        decomposition.

        Parameters
        ----------
        bw : float, optional
            The bandwidth of the kernel density estimation. Default is None and the bandwidth
            will be estimated using the Scott's rule of thumb.

        Returns
        -------
        clusters : dict

        """
        t_norm = self.coordinates["normalized distance"]

        if bw is None:
            """ Initial bandwidth estimation by Scott's rule """
            std = np.std(t_norm)
            bw = bw_scott(std, t_norm.size) / 5

        print("Initial bandwidth: {}".format(bw))

        """  Kernel density estimation   """
        t_test = np.linspace(0, 1, 1000)
        clusters = kdeScreen(t_norm, t_test, bw, plot=False)
        print("Number of clusters: {}".format(len(clusters['cluster centers'])))

        clusters["cluster centers"] = t_test[clusters["cluster centers"]]
        clusters["cluster boundary"] = t_test[clusters["cluster boundary"]]

        self.clusters = clusters

        if save_path is not None:
            save_path = save_path + ".stat" if not save_path.endswith(".stat") else save_path
            pk_save(save_path, clusters)

        return clusters

    def smooth_window(self, size, h_res, extend=2):
        """
        The window size of smoothing operation. The window size is the number of
        slices in the axial direction of fiber tow that are smoothed per iteration.

        A smaller window size will significantly decrease the computation time of
        smoothing operation.

        Parameters
        ----------
        size : int
            The window size.
        h_res : int
            The number of slices in the axial direction of fiber tow.
        extend : int, optional
            The number of slices that are extended to the left and right of the
            smoothing window to ensure the smoothness of the surface at the
            boundary of the windows.

        Returns
        -------
        wins_interp : ndarray
            The window size for interpolation (with extensions at the two ends).
        wins_result : ndarray
            The index of effective smoothing results (without extensions).
        """
        if size > 0.5 * h_res:
            print("The window size is too large to be processed as more than two groups. "
                  "The smoothing operation will be performed on the whole fiber tow.")
            return np.array([[0, h_res]]), np.array([[0, h_res]])

        n_intervals = int(h_res / size) + 1
        begin = (np.arange(0, size * n_intervals, size))[:-1]
        end = begin + size

        win_interp = []
        for i in np.arange(begin.size):
            beg_0 = begin[i] - extend
            en_0 = end[i] + extend

            win_interp.append([beg_0, en_0])

        win_interp = np.array(win_interp)
        win_interp[win_interp < 0] = 0

        win_result = np.zeros_like(win_interp)
        win_result[0, :] = [0, size]
        win_result[1:, :] = [extend, extend + size]
        win_result[-1, :] = [extend, extend + h_res - 1 - win_interp[-1, 0]]

        return win_interp, win_result

    def save(cls, save_path):
        """
        Save the fiber tow data.

        Parameters
        ----------
        save_path : str
            The path to save the fiber tow data.
        """
        # split the file name and path using os.path.split()
        path, filename = os.path.split(save_path)

        # if the path does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

        save_tow(save_path, cls)

    def axial_lines(self, save_path=None, plot=True):
        """
        Generate the axial line of the fiber tow surface.

        Parameters
        ----------
        save_path : str, optional
            The path to save the axial line data as a vtk mesh file.
        plot : bool, optional
            Whether to plot the axial line. Default is True.

        Returns
        -------
        axial_line : vtkPolyData
            The axial lines of the fiber tow.
        """
        # check if self._Tow__kriged_vertices exists
        if not hasattr(self, "_Tow__kriged_vertices"):
            pts_krig, _ = self.krig_cs()
        else:
            pts_krig = self._Tow__kriged_vertices[:, self.__column_order__]

        theta_res = int(self.theta_res)
        lines = pts_krig.reshape([-1, theta_res, 3])

        for i in range(theta_res):
            line = Curve(lines[:, i, :])
            cells = line.cells
            cells[:, 1:] += i * line.n_points

            point_data = np.full((line.n_points, 1), i, dtype=np.int_)

            try:
                line_set = np.vstack((line_set, line.points))
                cells_set = np.vstack((cells_set, cells))
                point_data_set = np.vstack((point_data_set, point_data))
            except NameError:
                line_set = line.points
                cells_set = cells
                point_data_set = point_data

        poly = pv.PolyData()
        poly.points = line_set
        poly.lines = cells_set
        poly.point_data["theta"] = point_data_set.ravel()

        if save_path is not None:
            path, filename = os.path.split(save_path)
            if not os.path.exists(path):
                os.makedirs(path)
            poly.save(save_path)

        if plot:
            poly.plot()

        # self.axial_line = poly
        return poly

    def radial_lines(self, save_path=None, plot=True, type="resampled"):
        """
        Generate the radial line of the fiber tow surface.

        Parameters
        -----------
        save_path : str, optional
            The path to save the radial line data as a vtk mesh file.
        plot : bool, optional
            Whether to plot the radial line. Default is True.
        type : str, optional
            The type of radial line. Default is "resampled". The other option is
            "original".

        Returns
        -------
        radial_line : vtkPolyData
            The radial lines of the fiber tow.
        """
        if type == "resampled":
            # check if self._Tow__kriged_vertices exists
            if not hasattr(self, "_Tow__kriged_vertices"):
                lines, _ = self.krig_cs()
            else:
                lines = self._Tow__kriged_vertices

        elif type == "original":
            lines = self.coordinates.iloc[:, [-3,-2,-1]].values

        slices = np.unique(lines[:, -1])

        n=0
        for i in slices:
            mask = lines[:, -1] == i
            line = Curve(lines[mask])
            cells = line.cells
            cells[:, 1:] += n

            n += line.n_points

            point_data = np.full((line.n_points, 1), i, dtype=np.int_)

            try:
                line_set = np.vstack((line_set, line.points))
                cells_set = np.vstack((cells_set, cells))
                point_data_set = np.vstack((point_data_set, point_data))
            except NameError:
                line_set = line.points
                cells_set = cells
                point_data_set = point_data

        poly = pv.PolyData()
        poly.points = line_set[:, self.__column_order__]
        poly.lines = cells_set
        poly.point_data["h_res"] = point_data_set.ravel()

        if save_path is not None:
            path, filename = os.path.split(save_path)
            if not os.path.exists(path):
                os.makedirs(path)
            poly.save(save_path)

        if plot:
            poly.plot(show_axes=None)

        # self.radial_line = poly
        return poly

    def normal_cross_section(self, algorithm="kriging", save_path=None,
                             plot=True, i_size=0.7, j_size=1, skip=10):
        """
        Generate the normal cross section of the fiber tow surface.

        Parameters
        ----------
        algorithm : str, optional
            The algorithm to generate the cross section. Default is "kriging". The other
            option is "pyvista" which uses pyvista's mesh clip function (Polydata.clip_surface()).
        save_path : str, optional
            The path to save the normal cross sections as a vtk mesh file.
        plot : bool, optional
            Whether to plot the normal cross sections. Default is True.
        i_size : float, optional
            The size of the i direction of the noraml plane. Default is 0.7.
        j_size : float, optional
            The size of the j direction of the noraml plane. Default is 1.
        skip : int, optional
            The number of cross sections to skip in plot. Default is 10. If skip is 1, all
            cross sections will be plotted.

        Returns
        -------
        cross_section : pyvista.PolyData
            The normal cross section of the fiber tows stored in a pyvista.PolyData object.
            Note that only the cross sections that are plotted are stored. If one wants to
            save all the cross sections, set skip=1.
        planes : pyvista.PolyData
            The corresponding planes that the cross sections are generated from. Note that
            only the planes that are plotted are stored. If one wants to save all the planes,
            set skip=1.
        """
        if algorithm == "pyvista":
            mesh = self.surf_mesh(plot=False, save_path=None)

            points = mesh.points
            cells = get_cells(mesh)

            s1 = pv.PolyData()
            s1.points = points
            s1.faces = cells
            s1 = s1.triangulate()

            trajectory = self.__traj
            direction = self.orientation

            cross_section = pv.PolyData()
            planes = pv.PolyData()
            pl = pv.Plotter()
            _ = pl.add_mesh(s1, style='wireframe', color='black', opacity=0.2)

            area = []
            perimeter = []
            width = []
            height = []
            for i in np.arange(0, trajectory.shape[0]):
                p = pv.Plane(center=trajectory[i],
                             direction=direction[i],
                             i_size=i_size, j_size=j_size).triangulate()
                p.point_data.clear()

                clipped = p.clip_surface(s1, invert=False)

                # sort the points on the boundary of clipped mesh
                edges = clipped.extract_feature_edges(30)
                edge = np.array(get_cells(edges))

                edge_reorder = edge[0, :]

                n_iter = edge.shape[0]
                for j in range(1, n_iter):
                    # find the index where the row contains the last element of edge_reorder
                    index = np.where(edge[1:, 1:] == edge_reorder.flatten()[-1])[0][0] + 1
                    edge_reorder = np.vstack((edge_reorder, edge[index, :]))
                    edge = np.delete(arr=edge, obj=index, axis=0)

                connectivity = edge_reorder[:, 1]
                points_boundary = edges.points[connectivity]

                polygon = Polygon(points_boundary)

                centroid = trajectory[i]
                points_boundary_local = points_boundary - centroid
                radius = np.sort(np.linalg.norm(points_boundary_local, axis=1))

                # height of the cross section
                h = np.average(radius[:4]) * 2
                # width of the cross section
                wid = np.average(radius[-4:]) * 2

                area.append(clipped.area)
                perimeter.append(polygon.perimeter)
                width.append(wid)
                height.append(h)

                if i % skip == 0:
                    cross_section = cross_section.merge(clipped)
                    planes = planes.merge(p)

                    _ = pl.add_mesh(p, style='surface', opacity=0.5)
                    _ = pl.add_mesh(clipped, color='r', line_width=10)

            if plot:
                _ = pl.show()

            if save_path is not None:
                path, filename = os.path.split(save_path)
                if not os.path.exists(path):
                    os.makedirs(path)
                elif not save_path.endswith(".vtk"):
                    save_path += ".vtk"

                cross_section.save(save_path)

            # update the geometry information
            self.geom_features["Area"] = area
            self.geom_features["Perimeter"] = perimeter
            self.geom_features["Width"] = width
            self.geom_features["Height"] = height

            return cross_section, planes, clipped

