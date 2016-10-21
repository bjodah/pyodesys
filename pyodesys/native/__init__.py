"""
pyodesys.native
===============
This module contains a SymbolicSys subclass which can be used to generate
source code representing the system of equations in a native language.

For some integrators (Sundials and GSL), the compiled objects from the
generated code are dynamically linked against respective library. You
will need to match the choice of BLAS/LAPACK implementation to link against.
This can be modified by setting the environment variables ``LBLAS`` and ``LLAPACK``
respectively.
"""
