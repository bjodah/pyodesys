#!/usr/bin/env python3
"""Use this utility for symbolic treatment.

Example usage:

$ python3 compensated_cse.py demo1 | clang-format --style=Google | batcat -pl C

"""
import pprint
from collections import defaultdict
from functools import reduce
from operator import add
from sympy import (
    Abs, Add, Basic, cse, exp, Expr, numbered_symbols, Piecewise, pi,
    postorder_traversal, preorder_traversal, pycode, Symbol, Tuple
)
from sympy.codegen import Assignment, aug_assign, CodeBlock
from sympy.codegen.ast import Token, Variable, float64, value_const

from .core import NullTransformer
from .util import OrderedAdd

class _NeumaierAdd(Token, Expr):
    """Represents KBN compensated summation."""

    __slots__ = ('terms', 'accum', 'carry', 'temp')
    _construct_terms = staticmethod(lambda args: Tuple(*args))

    def _ccode(self, printer):
        terms = ", ".join(map(printer._print, self.terms))
        return f"NA({terms} /*{str(self.accum)[:-1]}*/)"

    def to_statements(self, existing, expanded):
        """Transform into statements."""
        neum, ordinary = [], []
        for term in self.terms:
            if term in existing:
                neum.append(existing[term])
            else:
                ordinary.append(term)
        st = []
        if neum:
            st.append(Assignment(self.accum, sum(na.accum for na in neum)))
            st.append(Assignment(self.carry, sum(na.carry for na in neum)))
            for na in neum:
                expanded.add(na)
        else:
            st.append(Assignment(self.accum, ordinary.pop(0)))
            st.append(Assignment(self.carry, 0))

        for elem in ordinary:
            st.extend(_NeumaierAdd._impl_add(self.accum, self.carry, elem, self.temp))
        expanded.add(self)
        return st

    def finalize(self):
        """Close the summation."""
        return self._impl_finalize(self.accum, self.carry)

    @staticmethod
    def _impl_add(accum, carry, elem, temp):
        """Perform Kahan-Babuska-Neumaier addition."""
        big_temp = OrderedAdd(OrderedAdd(accum, -temp), elem)
        big_elem = OrderedAdd(OrderedAdd(elem, -temp), accum)
        pw = Piecewise((big_temp, Abs(temp) > Abs(elem)), (big_elem, True))
        return [
            Assignment(temp, accum + elem),
            aug_assign(carry, '+', pw),
            Assignment(accum, temp)
        ]

    @staticmethod
    def _impl_finalize(accum, carry):
        return Add(accum, carry)


class _NeumaierTransformer(NullTransformer):
    """Transform Add instances in CSEs to use compensated sum.

    Parameters
    ----------
    up_to: int, [0-100]
        Code is guaranteed to compile at levels 0 (no passes, no compensation)
        and 100 (all passes).
    """

    def __init__(self, repl, red, *, tmp_pfx="t", neu_pfx="n", up_to=100, limit=3, parent=None, ignore=None):
        self.repl = repl
        self.red = red
        self.limit = limit
        self.parent = parent
        self.ignore = ignore

        self.created = {}
        self._all_accum = {}
        self._all_carry = {}
        self._all_tempv = {}
        self.expanded = set()
        self._analysis = defaultdict(int)
        self._tmp_var = numbered_symbols(tmp_pfx)
        self._neu_var = numbered_symbols(neu_pfx)
        self.passes = []
        for p in filter(lambda n: n.startswith('_pass_'), dir(self)):
            null, rest = p.split("_pass_")
            assert null == ""
            num, *_ = rest.split("_")
            if len(_) == 0:
                continue
            if int(num) <= up_to:
                self.passes.append(getattr(self, p))

        self.statements, self.final_exprs = self._pipeline()

    def remapping_for_arrayification(self, template="m_glob[{0}]"):
        remapping = {}
        i = 0
        for st in self.statements:
            if st.lhs in remapping:
                continue
            if st.lhs in self._all_accum or st.lhs in self._all_carry:
                remapping[st.lhs] = Symbol(template.format(i), real=True)
            elif st.lhs in self._all_tempv:
                pass
            else:
                remapping[st.lhs] = Symbol(template.format(i), real=True)
            i = i + 1
        return remapping

    def _mk_Neu(self, terms, lhs):
        pfx = str(next(self._tmp_var)) if lhs is None else str(lhs)
        accum = Symbol(pfx+'a')
        carry = Symbol(pfx+'c')
        tempv = Symbol(pfx+'t')
        na = _NeumaierAdd(terms, accum, carry, tempv)
        self._all_accum[accum] = na
        self._all_carry[carry] = na
        self._all_tempv[tempv] = na
        return na

    @staticmethod
    def _is_Neu(x):
        return isinstance(x, _NeumaierAdd)

    def _pipeline(self):
        statements = [Assignment(*lr) for lr in self.repl]
        final_exprs = self.red
        for pass_ in self.passes:
            new_stmts, new_exprs = [], []
            for st in statements:
                new_rhs = pass_(st.lhs, st.rhs, statements=new_stmts)
                if st.lhs not in self.created:  # not Neumaier, (aug)assign:
                    new_stmts.append(st.__class__(st.lhs, new_rhs))
            for expr in final_exprs:
                new_exprs.append(pass_(None, expr, statements=new_stmts))
            statements = new_stmts
            final_exprs = new_exprs
        return statements, final_exprs

    def _pass_05_analysis(self, lhs, rhs, *, statements):
        if rhs.is_Add:
            if lhs is not None:
                self._analysis[lhs] += len(rhs.args)
            for term in rhs.args:
                if term in self._analysis:
                    self._analysis[term] += len(rhs.args) - 1
        return rhs

    def _pass_10_create_nodes(self, lhs, rhs, *, statements, debug=False):
        new_rhs = rhs
        if self.ignore:
            if lhs in self.ignore or any(rhs.has(ig) for ig in self.ignore):
                return new_rhs

        while True:
            for _add in filter(lambda x: x.is_Add, postorder_traversal(new_rhs)):
                score = self._analysis.get(lhs, 0) + reduce(add, [self._analysis.get(k, 1) for k in _add.args])
                if score >= self.limit or any(self._is_Neu(arg) for arg in _add.args):
                    na = self._mk_Neu(_add.args, lhs)
                    if _add is rhs and lhs is not None:
                        key = lhs

                    else:
                        key = next(self._neu_var)
                    self.created[key] = na
                    new_rhs = new_rhs.xreplace({_add: key})
                    break
            else:
                return new_rhs

        assert False

    def _pass_50_to_stmnts(self, lhs, rhs, *, statements):
        for arg in postorder_traversal(rhs):
            if arg in self.created:
                for t in list(self.created[arg].terms)+[arg]:
                    if t in self.created:
                        if arg not in self.expanded:
                            statements.extend(self.created[t].to_statements(self.created, self.expanded))
        return rhs

    def _pass_60_xrepl(self, lhs, rhs, *, statements):
        return rhs.xreplace(self.created)

    def _group(self, x):
        all_accum, all_carry = [], []
        for term in x.args:
            if term in self.created:
                all_accum.append(self.created[term].accum)
                all_carry.append(self.created[term].carry)
            elif term in self._all_accum:
                all_accum.append(term)
            elif term in self._all_carry:
                all_carry.append(term)
            else:
                all_accum.append(term)
        if all_accum and all_carry:
            return OrderedAdd(OrderedAdd(*all_accum), OrderedAdd(*all_carry))
        else:
            return OrderedAdd(*(all_accum+all_carry))

    def _pass_95_group(self, lhs, rhs, *, statements):
        return rhs.replace(lambda s: s.is_Add and any(
            t in self.created or t in self._all_accum or t in self._all_carry for t in s.args
        ), self._group)

    def _pass_90_fin(self, lhs, rhs, *, statements):
        return rhs.replace(lambda x: self._is_Neu(x), lambda x: x.finalize())


def _compensated_code(case, **kwargs):
    repl, red = cse(case.exprs)
    nm = _NeumaierTransformer(repl, red, **kwargs)
    statements = nm.statements_with_declarations()
    for i, new_expr in enumerate(nm.final_exprs):
        statements.append(Assignment(Symbol("out[%d]" % i), new_expr))
    return CodeBlock(*statements)
