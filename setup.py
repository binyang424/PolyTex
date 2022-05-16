from setuptools import setup

setup(
    name='polykriging',
    version='0.0.1',    
    description='A kriging package as a result of the course MEC 6310',
    url='https://github.com/shuds13/polykriging',
    author='Bin Yang',
    author_email='bin.yang@polymtl.ca',
    license='BSD 2-clause',
    packages=['polykriging'],
    install_requires=['numpy>=0.5',
                      'matplotlib',
                      'sympy',
                      'tk',             #tkinter
                      'shapely',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
