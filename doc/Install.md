# Installation

There are two supported ways to install PyImageJ: via [conda](https://conda.io/)/[mamba](https://mamba.readthedocs.io/) or via [pip](https://packaging.python.org/guides/tool-recommendations/). Although both tools are great [for different reasons](https://www.anaconda.com/blog/understanding-conda-and-pip), if you have no strong preference then we suggest using mamba because it will manage PyImageJ's non-Python dependencies [OpenJDK](https://en.wikipedia.org/wiki/OpenJDK) (a.k.a. Java) and [Maven](https://maven.apache.org/). If you use pip, you will need to install those two things separately.

## Installing via conda/mamba

Note: We strongly recommend using
[Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) rather
than plain Conda, because Conda is unfortunately terribly slow at configuring
environments.

1. [Install Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

2. Install PyImageJ into a new environment:
   ```
   mamba create -n pyimagej pyimagej openjdk=8
   ```

   This command will install PyImageJ with OpenJDK 8. If you would rather use
   OpenJDK 11, you can write `openjdk=11` instead, or even just leave off the
   `openjdk` part altogether to get the latest version of OpenJDK. PyImageJ
   has been tested most thoroughly with OpenJDK 8, but it is also known to
   work with OpenJDK 11, and likely later OpenJDK versions as well.

3. Whenever you want to use PyImageJ, activate its environment:
   ```
   mamba activate pyimagej
   ```

## Installing via pip

If installing via pip, we recommend using a
[virtualenv](https://virtualenv.pypa.io/) to avoid cluttering up or mangling
your system-wide or user-wide Python environment. Alternately, you can use
mamba just for its virtual environment feature (`mamba create -n pyimagej
python=3.8; mamba activate pyimagej`) and then simply `pip install` everything
into that active environment.

There are several ways to install things via pip, but we will not enumerate
them all here; these instructions will assume you know what you are doing if
you chose this route over conda/mamba above.

1. Install [Python 3](https://python.org/). As of this writing, PyImageJ has
   been tested with Python 3.6, 3.7, 3.8, 3.9, and 3.10.
   You might have issues with Python 3.10 on Windows.

2. Install OpenJDK 8 or OpenJDK 11. PyImageJ should work with whichever
   distribution of OpenJDK you prefer; we recommend
   [Zulu JDK+FX 8](https://www.azul.com/downloads/zulu-community/?version=java-8-lts&package=jdk-fx).
   Another option is to install openjdk from your platform's package manager.

3. Install pyimagej via pip:
   ```
   pip install pyimagej
   ```

## Testing your installation

Here's one way to test that it works:
```
python -c 'import imagej; ij = imagej.init("2.5.0"); print(ij.getVersion())'
```
Should print `2.5.0` on the console.