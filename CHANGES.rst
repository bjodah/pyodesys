v0.5.0
======
- OdeSys.solve() changed signature: first arg "solver" moved to kwargs and
  renamed "integrator". Default of None assumed (inspects $PYODESYS_INTEGRATOR)
- OdeSys.integrate_* renamed ``_integrate_*`` (only for internal use).
- Info dict from integrate() keys renamed (for consistency with pyneqsys):
    - nrhs -> nfev
    - njac -> njev
    - internal_xout (new)
    - internal_yout (new)

v0.4.0
======
- SymbolicSys not available directly from pyodesys (but from pyodesys.symbolic)
- OdeSys.integrate_* documented as private (internal).
- symbolic.PartiallySolvedSystem added
- multiple (chained) pre and postprocessors supported
- stiffness may be inspected retroactively (ratio biggest/smallest eigenvalue 
  of the jacobian matrix).

v0.3.0
======
- OdeSys.integrate* methods now return a tuple: (xout, yout, info-dict)
  currently there are no guarantees about the exact contents of the info-dict.
- signature of callbacks of rhs and jac in OdeSys are now:
      (t, y_arr, param_arr) -> f_arr
- two new methods: adaptive and predefined (incl. tests)
- Support roots
- Refactor plot_result (interpolation now available)
- Make Matrix class optional
- Added force_predefined kwarg to integrate()
- Fix bug in symmetricsys().from_callback()
- New upstream versions of pyodeint, pycvodes and pygslodeiv2
- Tweak tests of pycvodes backend for new upstream
- New example

v0.2.0
======
- New OdeSys class factory: symmetricsys for symmetric transformations
- Breaking change (for consistency with symneqsys): (lband, uband) -> band
- New convenience method: OdeSys.plot_result

v0.1.2
======
- added util.check_transforms

v0.1.1
======
- Variable transformations supported
- Only require sympy, numpy and scipy in requirements.txt

v0.1
====
- support for scipy, pyodeint, pygslodeiv2, pycvodes
