#!/usr/bin/env python
try:
    from numpy.distutils.core import Extension as NumpyExtension
    from numpy.distutils.core import setup

    from distutils.extension import Extension
    from Cython.Build import cythonize

    import numpy

except ImportError:
    raise ImportError('Numpy needs to be installed or updated.')


extensions = [
    NumpyExtension(
        name='BayHunter.surfdisp96_ext',
        sources=['src/extensions/surfdisp96.f'],
        extra_f77_compile_args='-O3 -ffixed-line-length-none -fbounds-check -m64'.split(),  # noqa
        f2py_options=['only:', 'surfdisp96', ':'],
        language='f77'),
    ]

extensions.extend(cythonize(
    Extension("BayHunter.rfmini", [
        "src/extensions/rfmini/rfmini.pyx",
        "src/extensions/rfmini/greens.cpp",
        "src/extensions/rfmini/model.cpp",
        "src/extensions/rfmini/pd.cpp",
        "src/extensions/rfmini/synrf.cpp",
        "src/extensions/rfmini/wrap.cpp",
        "src/extensions/rfmini/fork.cpp"],
        include_dirs=[numpy.get_include()])))


setup(
    name="pybinv",
    version="1.0",
    author="Cheng Song",
    author_email="songcheng@snu.ac.kr",
    description=("Parallel Tempering Transdimensional Bayesian Inversion of HVSR, RF, and/or SWD."),
    install_requires=[],
    url="",
    packages=['pybinv'],
    package_dir={
        'pybinv': 'src'},

    scripts=['src/scripts/baywatch'],

    package_data={
        'pybinv': ['defaults/*'], },

    ext_modules=extensions
)
