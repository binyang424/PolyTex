from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='polytex',
    version='0.4.5',
    description='PolyTex is an open-source toolkit for geometry modeling of fiber tow-based textiles based on volumetric images.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/binyang424/polytex',
    author='Bin Yang',
    author_email='bin.yang@polymtl.ca',
    # packages=['src', 'src.geometry',
    #           'src.kriging', 'src.mesh', 'src.plot',
    #           'src.stats', 'src.thirdparty'],
    packages=find_packages(),
    install_requires=['numpy==1.23',
                      'pyvista==0.39',
                      'pandas'
                      'matplotlib',
                      'sympy',
                      'tk',             #tkinter
                      'shapely',
                      'SciencePlots', 
                      'opencv-python',
                      'meshio',
                      'scikit-learn',
                      'opencv-python',
                      'tqdm'
                      ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
