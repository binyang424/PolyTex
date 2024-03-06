# `PolyTex`: A parametric textile geometry modeling package

## About this project

`PolyTex` is an open-source toolkit for geometry modeling of woven textiles based on volumetric images. It provides functionality such as geometrical feature extraction, local variability analysis and textile geometry modeling. A meshing module was implemented to generate voxel meshes. Generation of tetrahedral conformal meshes will be implemented in future release. Local material properties are assigned to each cell in the generated mesh, such that the anisotropic and heterogeneity are reflected. This image-based model is commonly referred to as a "Digital Material Twin". The toolkit is designed to provide material scientists with accurate numerical models to predict composite behaviors while not requiring extensive experience in image processing and mesh generation. Hence, Application programming interface (API) for `OpenFOAM` and `Abaqus` is provided.

We release this toolbox as an open-source project aiming to facilitate the application of numerical simulations based on digital material twins to engineering problems. In this regard, the project is well documented ([https://polytex.readthedocs.io/](https://polytex.readthedocs.io/)) and we would appreciate any contributions from the community (e.g. comments, suggestions, and corrections aimed at improving the software and documentation). 

Our issue tracker is at [https://github.com/binyang424/PolyTex/issues](https://github.com/binyang424/PolyTex/issues). Please report any bugs that you find or fork the repository on GitHub and create a pull request. We welcome all changes, big or small, and we will help you make the pull request if you are new to git.

> `PolyTex` is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
>
> `PolyTex` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See  [LICENSE](https://github.com/binyang424/PolyTex/blob/master/LICENSE.rst) for more details.

## Installation

To install `PolyTex` using `PyPI`, run the following command:

```shell
$ pip install polytex
```

To install `PolyTex` from the source, begin by cloning the repository using `git`:

```shell
$ git clone https://github.com/binyang424/PolyTex.git
```

> If you are unfamiliar with Git, you can refer to our tutorial [Git for beginners](https://github.com/binyang424/Git-for-beginners). Alternatively, you can download a specific branch of the source code from GitHub as a `.zip` file at https://github.com/binyang424/PolyTex.

Once the repository is cloned, navigate to the root directory of the `PolyTex` repository where the `setup.py` file is located, and execute the following command:

```bash
$ python setup.py install
```

To install `PolyTex` using the wheel file, navigate to the subdirectory `./dist/` of the downloaded `PolyTex` repository, and run:

```shell
$ pip install polytex-<version>.whl
```

## Contributing to `PolyTex`

Thank you for considering contributing to `PolyTex`! This project thrives on community contributions, and we appreciate your help.

### How Can You Contribute?

-   **Reporting Bugs:** If you find a bug, please open an issue at [https://github.com/binyang424/PolyTex/issues](https://github.com/binyang424/PolyTex/issues). Provide as much detail as possible, including your environment, steps to reproduce, and the expected vs. actual behavior.
    
-   **Suggesting Enhancements:** Have an idea for a new feature or an improvement? Feel free to open an issue and discuss it with the community.
    
-   **Code Contributions:** If you want to contribute code to `PolyTex`, fork the repository, create a new branch for your changes, and submit a pull request. Follow the coding standards and make sure your changes are well-tested.

### Getting Started

1.  Fork the `PolyTex` repository.
    
2.  Clone your forked repository to your local machine:
    
    ```shell
    $ git clone https://github.com/your-username/PolyTex.git
    ```
    
3.  Create a new branch for your changes:
    
    ```shell
    $ git checkout -b feature/your-feature
    ```
    
4.  Make your changes and commit them:
    
    ```shell
    $ git add .
    $ git commit -m "Add your commit message here"
    ```
    
5.  Push your changes to your fork:
    
    ```shell
    $ git push origin feature/your-feature
    ```
    
6.  Open a pull request on the `PolyTex` repository.
    

### Code Style

Follow the established code style in the project. Make sure your code is well-documented and includes tests when applicable. See [style guide](https://numpydoc.readthedocs.io/en/latest/format.html) for docstrings used with the `numpydoc` extension for [Sphinx](https://www.sphinx-doc.org/).

### License
By contributing to `PolyTex`, you agree that your contributions will be licensed under the project's [LICENSE](https://github.com/binyang424/PolyTex/blob/master/LICENSE.rst).


## Citation

To cite `PolyTex` in publications use:

- Bin YANG, Yuwei Feng, Cédric BÉGUIN, Philippe CAUSSE and Jihui WANG. Open Source Tool for Micro-CT Aided Meso-scale Modeling and
  Meshing of Complex Textile Composite Structures. Submitted to *Composites Science and Technology* (2024).
