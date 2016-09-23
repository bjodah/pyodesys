#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import shutil
from setuptools import setup


pkg_name = 'pyodesys'

RELEASE_VERSION = os.environ.get('PYODESYS_RELEASE_VERSION', '')  # v*

# http://conda.pydata.org/docs/build.html#environment-variables-set-during-the-build-process
if os.environ.get('CONDA_BUILD', '0') == '1':
    try:
        RELEASE_VERSION = 'v' + open(
            '__conda_version__.txt', 'rt').readline().rstrip()
    except IOError:
        pass


def _path_under_setup(*args):
    return os.path.join(os.path.dirname(__file__), *args)

release_py_path = _path_under_setup(pkg_name, '_release.py')

if len(RELEASE_VERSION) > 0:
    if RELEASE_VERSION[0] == 'v':
        TAGGED_RELEASE = True
        __version__ = RELEASE_VERSION[1:]
    else:
        raise ValueError("Ill formated version")
else:
    TAGGED_RELEASE = False
    # read __version__ attribute from _release.py:
    exec(open(release_py_path).read())

classifiers = [
    "Development Status :: 3 - Alpha",
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
]

submodules = [
    'pyodesys.native',
]

tests = [
    'pyodesys.tests',
    'pyodesys.native.tests',
]

with open(_path_under_setup(pkg_name, '__init__.py'), 'rt') as f:
    short_description = f.read().split('"""')[1].split('\n')[1]
assert 10 < len(short_description) < 80
long_description = io.open(_path_under_setup('README.rst'),
                           encoding='utf-8').read()
assert len(long_description) > 100

setup_kwargs = dict(
    name=pkg_name,
    version=__version__,
    description=short_description,
    long_description=long_description,
    classifiers=classifiers,
    author='Björn Dahlgren',
    author_email='bjodah@DELETEMEgmail.com',
    url='https://github.com/bjodah/' + pkg_name,
    license='BSD',
    packages=[pkg_name] + submodules + tests,
    include_package_data=True,
    extras_require={
        'all': ['appdirs', 'dill', 'sym', 'sympy', 'scipy', 'pyodeint',
                'pycvodes', 'pygslodeiv2', 'pycompilation', 'pycodeexport'],
        'docs': ['Sphinx', 'sphinx_rtd_theme', 'numpydoc']
    }
)

if __name__ == '__main__':
    try:
        if TAGGED_RELEASE:
            # Same commit should generate different sdist
            # depending on tagged version (set $PYODESYS_RELEASE_VERSION)
            # e.g.:  $ PYODESYS_RELEASE_VERSION=v1.2.3 python setup.py sdist
            # this will ensure source distributions contain the correct version
            shutil.move(release_py_path, release_py_path+'__temp__')
            open(release_py_path, 'wt').write(
                "__version__ = '{}'\n".format(__version__))
        setup(**setup_kwargs)
    finally:
        if TAGGED_RELEASE:
            shutil.move(release_py_path+'__temp__', release_py_path)
