v0.12.2
=======
- cvodes code now support multiple precisions (sundials may be compiled with float, double or long double)

v0.12.1
=======
- Fix CSE in native code-generation

v0.12.0
=======
- Support for "jtimes" (jacobian times vector) for use with iterative solvers
- NativeSys now uses latest AnyODE

v0.11.16
========
- Fix CSE in native code-generation

v0.11.15
========
- Updated (upper) requirements of pycvodes, pygslodeiv2, pyodeint (v0.12.0 will again require latest versions)

v0.11.14
========
- Relax a few native tests

v0.11.13
========
- Further fixes to only require C++11

v0.11.12
========
- Change C++ requirement to C++11

v0.11.11
========
- Bumpy AnyODE to version 14

v0.11.10
========
- Bump AnyODE to version 13

v0.11.9
=======
- Bump AnyODE to version 12

v0.11.8
=======
- Certain gcc versions giving trouble with ``std::abs``, switching to C's ``fabs``.

v0.11.7
=======
- chained_parameter_variation now also available as a method.

v0.11.6
=======
- Changed install_requirements in setup.py to be less minimalistic.
- Improved install instructions.

v0.11.5
=======
- Introduced ODESys.numpy to allow user to used a patched numpy module. (e.g. for quantitites)

v0.11.4
=======
- Make sure nonlinear_invariants are copied in SymbolicSys.from_other.

v0.11.3
=======
- Harden a intermittently failing test (dict ordering)

v0.11.2
=======
- to_arrays now accepts a kwarg 'reshape'
- New (convenience) methods for SymbolicSys:
    - from_other_new_params
    - from_other_new_params_by_name

v0.11.1
=======
- Fixes to to_arrays (in combination with as_autonomous & from_other)
- chained_parameter_variation got a new parameter: npoints
- latex_names in plotting are now assumed not to be wrapped in $$.

v0.11.0
=======
- Using new upstream pycvodes-0.9.0 (new ABI for native_sys['cvode'])

v0.10.4
=======
- Optimized away bottleneck for back transforming data.

v0.10.3
=======
- Fix for scalar bounds in SymbolicSys
  
v0.10.2
=======
- Softer bounds checking
- with_jtimes (pycvodes >= 0.8.3)

v0.10.1
=======
- Bump AnyODE

v0.10.0
=======
- Added SymbolicSys.as_autonomous()
- SymbolicSys arg "params" now need to be ``True`` to induce deduction.

v0.9.2
======
- Copy pyx to cache dir prior to cythonize

v0.9.1
======
- Address ``indep`` by name
- Bumpy AnyODE

v0.9.0
======
- Support for max_invariant_violations
- Expose special settings
- Fix dropping units

v0.8.1
======
- Fix bug in PartiallySolvedSystem when passing linear_invariants to constructor.

v0.8.0
======
- New function ``core.integrate_chained`` for use with TransformedSys.
- Calls to ``f(x, y[:], p[:])`` now carries y0 in p[np:np+ny] (also applies to jac, etc.)
- Renamed OdeSys to ODESys (OdeSys left as a deprecated alias)
- New arguments to ODESys: dep_by_name, par_by_name, param_names, latex_names, latex_param_names
- New kwargs: first_step_{cb,expr,factory} in ODESys, SymbolicSys & SymbolicSys.from_callback respectively.
- SymbolicSys.jacobian_singular() returns bool (uses cse and LUdecomposition raising ValueError)
- New module: .results

v0.7.0
======
- Generate (multi-threaded) C++ code (against pyodeint, pycvodes, pygslodeiv2)
- OdeSys.internal_* -> OdeSys._internal

v0.6.0
======
- depend on package ``sym`` for symbolic backend

v0.5.3
======
- minor fixes

v0.5.2
======
- from_callback now respects backend paramter (e.g. ``math`` or ``sympy``)

v0.5.1
======
- Added SymbolicSys.analytic_stiffness
- Allow chaining pre-/post-processors in TransformedSys
- Make PartiallySolvedSys more general (allow use dependent variable)
- Better choice of first_step when not specified (still arbitrary though)
- Documentation fixes
- SymbolicSys got a new classmethod: from_other

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
