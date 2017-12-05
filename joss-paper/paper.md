---
title: 'pyodesys: Straightforward numerical integration of ODE systems from Python'
tags:
  - ordinary differential equations
  - symbolic derivation
  - symbolic transformations
  - code-generation
authors:
 - name: Björn Dahlgren
   orcid: 0000-0003-0596-0222
   affiliation: 1
affiliations:
 - name: KTH Royal Institute of Technology
   index: 1
date: 5 December 2017
bibliography: paper.bib
---

# Summary
The numerical integration of systems of ordinary differential equations (ODEs) is very
common in most scientific fields. There exists a large number of software libraries
for solving these systems, each requiring the user to write their code in slightly
different forms. Furthermore, it is sometimes necessary to perform variable transformations
in order for the solution to proceed efficiently.

*pyodesys* provides a unified interface to some existing solvers. It also provides an interface
to represent their system symbolically. This allows *pyodesys* to derive the Jacobian matrix
symbolically (which is both tedious and error prone done performed manually). In addition, this
symbolic representation allows the user to manipulate the mathematical representation symbolically. This is achieved
by using SymPy [@Meurer2017] (although the coupling is loose and other symbolic backends
may be used).

*pyodesys* enables the user to write his
or her code once, and adaptions to different libraries are handled internally. This allows
the user to evaluate both different solvers (which implement different integration methods and
algorithms for step size control) and alternative formulations of the system (from variable
transformations, including scaling of variables).


# Features
- Unified interface to ODE solvers from Sundials [@hindmarsh2005sundials],
  GNU Scientific Library [@galassi_gnu_2009] and odeint in boost [@Ahnert2011]
- Convenince methods for working with solutions (plotting trajectories, interpolation, inspecting invariants).
- Automatic derivation of the Jacobian matrix for use with implicit steppers.
- Symbolic variable transformations in the system of ODEs.
- Symbolic reduction of degrees of freedom by variable elimination using linear invariants.
- Symbolic rewriting of system based on (possibly approximate) analytic solutions to a subset of dependent variables.
- Code-generation (C++) and on-the-fly compilation for enhanced performance (common subexpression elimintation is
  automatically performed).
- Parallel execution for parameter variations. Multiple integrations may be performed in parallel
  (multithreading using OpenMP), this feature is only availble in conjuction with code-generation.

# References