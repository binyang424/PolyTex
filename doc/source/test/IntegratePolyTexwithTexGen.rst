.. Integrate ``PolyTex`` with ``TexGen``
.. =====================================

``TexGen`` software, developed at the `University of
Nottingham <https://www.nottingham.ac.uk/research/groups/composites-research-group/meet-the-team/louise.brown>`__,
is an excellent tool for building mesoscale models of textile
reinforcements and composites. ``PolyTex`` can be used as a third-party
package via Python 3.9 with the latest ``TexGen`` release (3.13.x). This
allows data and methods provided by ``PolyTex`` to be directly accessed
by ``TexGen`` to build ideal models of woven textiles and textile
composites. To do this, you should install the ``TexGen`` package
without Python bundled. A brief description is given below.

Step 1: Download ``TexGen``
---------------------------

Navigate to the `TexGen download page
on SourceForge.net <https://sourceforge.net/projects/texgen/>`__ and
follow the steps shown in the figure below. Note that you need to
download the release without Python bundled.

.. image:: ./images/texgen-download.png

The ``TexGen`` file (``texgen-Python39-3.13.1.exe``) indicates that
the current ``TexGen`` package supports only Python 3.9.

Step 2: Install Python 3.9 
---------------------------

Go to the official website of Python: `Download Python \|
Python.org <https://www.python.org/downloads/>`__ and select the
required Python version (Python 3.9). Here is the link to `Python
Release Python
3.9.13 <https://www.python.org/downloads/release/python-3913/>`__.

Step 3: Install ``TexGen``
--------------------------

Install the ``TexGen`` package downloaded in Step 1. ``TexGen`` will
automatically detect the installation of Python 3.9.

.. image:: ./images/image-20240624121115642.png

Step 4: Install ``PolyTex`` in the Python 3.9 Environment
---------------------------------------------------------

``PolyTex`` can be installed simply by executing:

.. code:: python

   pip install polytex==0.4.5

If multiple versions of Python are installed on your computer, specify
the Python 3.9 installation directory in the pip command using the
``--target=`` option:

.. code:: shell

   pip install polytex==0.4.5 --target=C:/Users/User/AppData/Local/Programs/Python/Python39/Lib/site-packages

Once the installation is complete, import ``PolyTex`` in the ``TexGen``
Python console:

.. code:: python

   import polytex

To test if the installation is successful, use the following code:

.. code:: python

   polytex.__author__
   polytex.__version__

This should return the developer and version information of the
installed ``PolyTex``.

.. image:: ./images/test_installation.png

At this point, all functions and data in ``PolyTex`` can be accessed by
``TexGen`` to build models. For different fabric structures, different
``TexGen`` scripts are required. Examples are provided in the ``TexGen``
repository `louisepb/TexGenScripts: Sample scripts to demonstrate TexGen
scripting <https://github.com/louisepb/TexGenScripts>`__.

Step 5: Convert ``TexGen`` voxel model to ``OpenFOAM`` mesh
-----------------------------------------------------------

Geometrical models created by ``TexGen`` can be exported as a `VTU Voxel
File <https://texgen.sourceforge.io/index.php/User_Guide#VTU_Voxel_File>`__.
This file format can then be converted to ``OpenFOAM`` meshes using
``PolyTex``. An example script for this conversion process can be found
at `Convert the TexGen-generated vtu file to OpenFOAM polyMesh — PolyTex
documentation <https://polytex.readthedocs.io/en/latest/source/test/texgen_vtu_2_foam.html>`__.
