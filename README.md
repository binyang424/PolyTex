# `PolyKriging`: A parametric geometry modeling package

## About this project

`PolyKriging` is an open-source toolkit for geometry modeling of fiber tow-based textiles based on volumetric images. It provides functionality such as geometrical feature extraction, local variability analysis, modeling accuracy evaluation, and interface smoothing. A meshing module was implemented to generate tetrahedral conformal meshes and voxel meshes. Local material properties are assigned to each cell, such that the anisotropic and heterogeneity are reflected. This image-based model is commonly referred to as a "Digital Material Twin". The toolkit is designed to provide material scientists with accurate numerical models to predict composite behaviors while not requiring extensive experience in image processing and mesh generation.

We release this toolbox as an open-source project aiming to facilitate the application of numerical simulations based on digital material twins to engineering problems. In this regard, the project is well documented and we would appreciate any contributions from the community (e.g. comments, suggestions, and corrections aimed at improving the software and documentation). 

Our issue tracker is at [https://github.com/binyang424/polykriging/issues](https://github.com/binyang424/polykriging/issues). Please report any bugs that you find or fork the repository on GitHub and create a pull request. We welcome all changes, big or small, and we will help you make the pull request if you are new to git.

> `PolyKriging` is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
>
> `PolyKriging` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [License](./LICENSE.html) for more details.

## Installation

**Note: This document is only a template and will be completed when the project is released. The following information may not work at this stage**.

To install polyKriging using PyPI, run the following command:

```
$ pip install polykriging
```

To install polyKriging using Anaconda, run the following command:

```
$ conda install -c anaconda polykriging
```

To install `PolyKriging` from GitHub source, first clone `PolyKriging` using `git`(check our tutorial [Git for beginners](https://github.com/binyang424/Git-for-beginners) if you are new to this):

```
$ git clone https://github.com/binyang424/polykriging.git
```

Then, in the `PolyKriging` repository that you cloned, simply run:

```
$ python setup.py install
```

## Contributing

We welcome contributions from anyone, even if you are new to open source. Please read our [Introduction to Contributing](https://www.binyang.fun/polykriging/Introduction-to-contributing) page and the [PolyKriging Documentation Style Guide](https://www.binyang.fun/polykriging/documentation-style-guide.html). If you are new and looking for some way to contribute, a good place to start is to look at the issues tagged [Easy to Fix](https://github.com/binyang424/polykriging/issues).


## Citation

- To cite `PolyKriging` in publications use:



- A BibTeX entry for LaTeX users is:

  
