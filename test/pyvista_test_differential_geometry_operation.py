import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import colorcet
import threading
from pyvista import examples
import multiprocessing


def mesh_cmp_custom(mesh, name):
    """
    自定义色彩映射
    :param mesh: 输入mesh
    :param name: 比较数据的名字
    :return:
    """
    pts = mesh.points
    mesh[name] = pts[:, 1]
    # Define the colors we want to use
    blue = np.array([12 / 256, 238 / 256, 246 / 256, 1])
    black = np.array([11 / 256, 11 / 256, 11 / 256, 1])
    grey = np.array([189 / 256, 189 / 256, 189 / 256, 1])
    yellow = np.array([255 / 256, 247 / 256, 0 / 256, 1])
    red = np.array([1, 0, 0, 1])

    c_min = mesh[name].min()
    c_max = mesh[name].max()
    c_scale = c_max - c_min

    mapping = np.linspace(c_min, c_max, 256)
    newcolors = np.empty((256, 4))
    newcolors[mapping >= (c_scale * 0.8 + c_min)] = red
    newcolors[mapping < (c_scale * 0.8 + c_min)] = grey
    newcolors[mapping < (c_scale * 0.55 + c_min)] = yellow
    newcolors[mapping < (c_scale * 0.3 + c_min)] = blue
    newcolors[mapping < (c_scale * 0.1 + c_min)] = black

    # Make the colormap from the listed colors
    my_colormap = ListedColormap(newcolors)
    mesh.plot(scalars=name, cmap=my_colormap)


def mesh_cmp_mpl(mesh, name):
    """
        使用Matplotlib进行色彩映射
        :param mesh: 输入mesh
        :param name: 比较数据的名字
        :return:
        """
    pts = mesh.points
    mesh[name] = pts[:, 1]
    mlp_cmap = plt.cm.get_cmap("viridis", 25)
    mesh.plot(scalars=name, cmap=mlp_cmap)


def mesh_cmp(mesh, name):
    """
       使用进行plot自带的色彩映射
       :param mesh: 输入mesh
       :param name: 比较数据的名字
       :return:
    """
    pts = mesh.points
    mesh[name] = pts[:, 1]
    mesh.plot(scalars=name, cmap='viridis_r')


def mesh_cmp_colorcet(mesh, name):
    """
       使用进行colorcet进行色彩映射
       :param mesh: 输入mesh
       :param name: 比较数据的名字
       :return:
    """
    pts = mesh.points
    mesh[name] = pts[:, 1]
    mesh.plot(scalars=name, cmap=colorcet.fire)


if __name__ == '__main__':
    # mesh = pv.read('vtkData/airplane.ply')
    mesh = examples.download_embryo()
    # 开启多进程用于可视化曲面
    p1 = multiprocessing.Process(target=mesh_cmp_custom, args=(mesh, 'y_height',))
    p2 = multiprocessing.Process(target=mesh_cmp_mpl, args=(mesh, 'y_height',))
    p3 = multiprocessing.Process(target=mesh_cmp, args=(mesh, 'y_height',))
    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
