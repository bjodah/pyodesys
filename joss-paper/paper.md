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
The numerical integration of systems of ordinary differential equations is very
common in most scientific fields. There exists a large number of software libraries
for solving these systems, each requiring the user to write their code in slightly
different forms. And sometimes variable transformations need to be applied for the
solution to proceed efficiently. *pyodesys* enables the user to write his or her code
once, and adaptions to different libraries are handled internally.

In addition to providing a unified interface to exisiting solvers, pyodesys allows
the user to manipulate the mathematical representation symbolically. This is achieved
by using SymPy [@Meurer2017] (although the coupling is loose and other symbolic backends
may be used).


# Features
- Unified interface to ODE solvers from Sundials [@hindmarsh2005sundials], GNU Scientific Library [@galassi_gnu_2009],
  odeint in boost [@Ahnert2011]
- Convenince methods for working with solutions (plotting trajectories, interpolation, inspecting invariants).
- Automatic derivation of the jacobian matrix for use with implicit steppers.
- Symbolic variable transformations of the system of ODEs.
- Symbolic reduction of degrees of freedom by variable elimination using linear invariants.
- Symbolic rewriting of system based on (possibly approximate) analytic solutions to a subset of dependent variables.
- Code-generation (C++) and on-the-fly compilation for enhanced performance (common subexpression elimintation is
  automatically performed).
- Parallel execution: for parameter variation multiple integrations may be performed in parallel
  (using OpenMP threading model), this feature is only availble in conjuction with code-generation.

# References