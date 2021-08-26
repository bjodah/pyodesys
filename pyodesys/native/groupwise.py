"""Handle common cses among groups of code."""

from functools import reduce
from operator import add
import numpy as np

from sympy import Symbol, cse, numbered_symbols
from sympy.codegen.ast import Assignment, Variable, float64
from sympy.codegen.rewriting import create_expand_pow_optimization
from sympy.printing.c import ccode


from .core import NullTransformer


expand_up_to_3 = create_expand_pow_optimization(3)


def pre_process(expr):
    """Simplify, expand & factor."""
    return expr.simplify(rational=True).expand().factor()


def post_process(expr):
    """Expand low integer powers."""
    return expand_up_to_3(expr)


class GroupwiseCSE:
    """Eliminate common sub-expressions from groups of expressions."""

    def __init__(self, groups, *,
                 common_cse_template="common_cse{0}",
                 common_ignore=(),
                 to_code=lambda arg: ccode(arg, math_macros={}),
                 subsd=None,
                 Transformer=NullTransformer,
                 type_=float64,
                 pre_process=pre_process,
                 post_process=post_process,
                 kw_cse=None
                 ):
        """
        Parameters
        ----------
        groups : list like
        \\*\\*kwargs : see code for now.

        """
        # self.groups = groups
        self._to_code = to_code
        self._subsd = subsd or {}
        self._type = type_
        self._keys, _values = zip(*groups.items())
        self._spans = np.cumsum([0]+list(map(len, _values)))
        self._common_cse_template = common_cse_template
        self._per_g_tformrs, self._comm_tformr = self._get_all_cses(
            map(pre_process, reduce(add, map(list, _values))),
            common_ignore=common_ignore,
            post_process=post_process,
            Transformer=Transformer,
            kw_cse=kw_cse or {},
        )

    @property
    def keys(self):
        """Retrieve the keys of the groups."""
        return self._keys

    def render(self, x):
        """Generate a code string."""
        return self._to_code(x.xreplace(self._subsd))

    def _common_cse(self, all_exprs, **kwargs):
        repls, reds = cse(all_exprs, **kwargs)
        # def comm_symbols():
        #     idx = 0
        #     while True:
        #         yield Symbol(self._common_cse_template.format(idx), real=True)
        #         idx += 1
        #cse_symbols = comm_symbols()
        cse_symbols = numbered_symbols("cse_t")  # local temporaries
        comm_subs = {}
        for lhs, rhs in repls:
            for expr in reds:
                if lhs in expr.free_symbols:
                    comm_subs[lhs] = next(cse_symbols)
                    break
        return (
            [(lhs.xreplace(comm_subs), rhs.xreplace(comm_subs)) for lhs, rhs in repls],
            [r.xreplace(comm_subs) for r in reds]
        )

    def _get_all_cses(self, all_exprs, *, common_ignore, post_process,
                      Transformer, kw_cse):
        repls, reds = self._common_cse(
            all_exprs, ignore=common_ignore,
            symbols=numbered_symbols('cse_temporary'), **kw_cse)
        comm_tformer = Transformer(repls, reds, ignore=common_ignore)

        assert(len(comm_tformer.final_exprs) == len(reds))
        del reds

        remap = comm_tformer.remapping_for_arrayification(template=self._common_cse_template)
        comm_tformer.apply_remapping(remap)
        self.n_remapped = len(remap)

        grp = {}
        for i, k in enumerate(self._keys):
            g_repls, g_exprs = cse(
                comm_tformer.final_exprs[slice(*self._spans[i:i+2])],
                symbols=numbered_symbols("cse"), **kw_cse
            )
            g_tformer = Transformer(g_repls, g_exprs, parent=comm_tformer)
            grp[k] = g_tformer

        return (grp, comm_tformer)

    @staticmethod
    def _declare(stmts, *, pred, type_=float64):
        seen = set()
        result = []
        for st in stmts:
            if isinstance(st, Assignment) and st.lhs not in seen and pred(st.lhs):
                seen.add(st.lhs)
                st = Variable(st.lhs, type=type_).as_Declaration(value=st.rhs)
            result.append(st)
        return result

    def common_statements(self, declare=False):
        """Initialize the common sub-expressions among the groups."""
        if declare:
            return self._comm_tformr.statements_with_declarations(pred=declare, type_=self._type)
        else:
            return self._comm_tformr.statements

    def statements(self, gk, declare=False):
        """Initialize the group specific sub-expressions."""
        if declare:
            return self._per_g_tformrs[gk].statements_with_declarations(pred=declare, type_=self._type)
        else:
            return self._per_g_tformrs[gk].statements

    def exprs(self, gk):
        """Retrieve the resulting expressions of the group named ``key``."""
        return self._per_g_tformrs[gk].final_exprs
