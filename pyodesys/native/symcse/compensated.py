#!/usr/bin/env python3
"""Use this utility for symbolic treatment.

Example usage:

$ python3 -m symcse demo-compensated-py
$ python3 -m symcse demo-compensated-c --index 3 | clang-format --style=Google | batcat -pl C

"""
from collections import defaultdict
from functools import reduce
from operator import add, mul
from sympy import (
    Abs,
    Add,
    And,
    Eq,
    Expr,
    Lt,
    Mul,
    Ne,
    numbered_symbols,
    Piecewise,
    postorder_traversal,
    Symbol,
    Tuple,
)
from sympy.codegen import Assignment, aug_assign, CodeBlock
from sympy.codegen.ast import AssignmentBase, Token, While, break_

from .core import NullTransformer
from .ordered_add import ordered_add


def If(cond, body):
    return While(cond, CodeBlock(*body, break_))

class _CompensatedAdd(Token, Expr):
    """Represents KBN compensated summation."""

    _fields = __slots__ = ("terms", "accum", "carry", "temp")
    _construct_terms = staticmethod(lambda args: Tuple(*args))

    def _ccode(self, printer):
        terms = ", ".join(map(printer._print, self.terms))
        return f"NA({terms} /*{str(self.accum)[:-1]}*/)"

    def to_statements(self, existing, expanded, *, transients, **kwargs):
        """Transform into statements."""
        neum, ordinary = [], []
        for term in self.terms:
            ex, other = [], []
            for factor in term.as_ordered_factors():
                if factor in existing:
                    ex.append(factor)
                else:
                    other.append(factor)
            if len(ex) == 1:  # TODO: handle >1
                neum.append((reduce(mul, other) if other else 1, existing[ex[0]]))
            else:
                ordinary.append(term)
        st = []
        if neum:
            if True:
                st.append(Assignment(self.accum, neum[0][0] * neum[0][1].accum))
                ordinary.extend([oth * na.accum for oth, na in neum[1:]])
            else:
                st.append(
                    Assignment(self.accum, sum(oth * na.accum for oth, na in neum))
                )

            st.append(Assignment(self.carry, sum(oth * na.carry for oth, na in neum)))
        else:
            st.append(Assignment(self.accum, ordinary.pop(0)))
            st.append(Assignment(self.carry, 0))

        for elem in ordinary:
            nops = elem.count_ops()
            if nops == 1 and isinstance(elem, Mul):
                nops = max(nops, (-elem).count_ops())
            if nops:
                # for fs in elem.free_symbols:
                #     if fs not in existing:
                #         continue
                #     ex = existing[fs]
                #     if ex not in expanded:
                #         continue
                #     e = elem.subs(fs, ex.accum + ex.carry)
                #     _accu, _pure_carry = [], []
                #     for term in e.as_ordered_terms():
                #         if ex.carry in term and ex.accum not in term:
                #             _pure_carry.append(term)
                #         else:
                #             _accum.append(term)
                #     if _pure_carry:
                #         ...
                #     ea = elem.subs(fs, ex.accum)
                #     ec = elem.sbus(fs, ex.carry)
                tr = next(transients)
                st.append(Assignment(tr, elem))
                elem = tr
            st.extend(
                self._impl_add(self.accum, self.carry, elem, self.temp, **kwargs)
            )
        expanded.add(self)
        return st

    def finalize(self):
        """Close the summation."""
        return self._impl_finalize(self.accum, self.carry)

    @staticmethod
    def _impl_finalize(accum, carry):
        return Add(accum, carry)

class _NeumaierAdd(_CompensatedAdd):
    _impl_add_kws = ('do_swap',)

    @staticmethod
    def _impl_add(accum, carry, elem, temp, do_swap=False):
        """Perform Kahan-Babuska-Neumaier addition."""
        big_temp = ordered_add(ordered_add(accum, -temp), elem)
        big_elem = ordered_add(ordered_add(elem, -temp), accum)
        abs_elem = Abs(elem)
        pw = Piecewise((big_temp, Abs(temp) > abs_elem), (big_elem, True))
        statements = [
            Assignment(temp, accum + elem),
            aug_assign(carry, "+", pw),
            Assignment(accum, temp),
        ]
        if do_swap:
            return [
                If(
                    And(Eq(carry, 0), Ne(accum, 0), Lt(Abs(accum), abs_elem)),
                    [
                        Assignment(temp, accum),
                        Assignment(accum, carry),
                        Assignment(carry, temp),
                    ],
                )
            ] + statements
        else:
            return statements


class _TwoSumAdd(_CompensatedAdd):
    _impl_add_kws = ()

    @staticmethod
    def _impl_add(accum, carry, elem, temp):
        """Perform 2Sum addition."""
        a_prim = ordered_add(temp, -elem)
        b_prim = ordered_add(temp, -a_prim)
        delta_a = ordered_add(accum, -a_prim)
        delta_b = ordered_add(elem, -b_prim)
        statements = [
            Assignment(temp, accum + elem),
            aug_assign(carry, "+", ordered_add(delta_a, delta_b)),
            Assignment(accum, temp)
        ]
        return statements


class _FastTwoSumAdd(_CompensatedAdd):
    _impl_add_kws = ()

    @staticmethod
    def _impl_add(accum, carry, elem, temp):
        """Perform Fast2Sum addition."""
        statements = [
            Assignment(temp, ordered_add(accum, elem)),
            aug_assign(carry, "+", ordered_add(elem, -ordered_add(temp, -accum))),
            Assignment(accum, temp),
        ]
        return statements


class _CompensationTransformer(NullTransformer):
    """Transform Add instances in CSEs to use compensated sum.

    Parameters
    ----------
    up_to_debug: int, [0-100]
        Code is guaranteed to compile at levels 0 (no passes, no compensation)
        and 100 (all passes).
    """

    _CompAdd = None  # Token

    def __init__(
        self,
        repl,
        red,
        *,
        tmp_pfx="t",
        trs_pfx="r",
        csum_pfx="n",
        up_to_debug=100,
        limit=3,
        parent=None,
        ignore=None,
        kw_to_statements=None
    ):
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
        self._trs_var = numbered_symbols(trs_pfx)
        self._csum_var = numbered_symbols(csum_pfx)
        self.passes = []
        for p in filter(lambda n: n.startswith("_pass_"), dir(self)):
            null, rest = p.split("_pass_")
            assert null == ""
            num, *_ = rest.split("_")
            if len(_) == 0:
                continue
            if int(num) <= up_to_debug:
                self.passes.append(getattr(self, p))

        self.kw_to_statements = kw_to_statements or {}
        self.statements, self.final_exprs = self._pipeline()

    def remapping_for_arrayification(self, template="m_glob[{0}]"):
        remapping = {}
        i = 0
        for st in self.statements:
            if st.lhs in remapping or st.lhs in self._all_tempv:
                continue
            # if st.lhs in self._all_accum or st.lhs in self._all_carry:
            remapping[st.lhs] = Symbol(template.format(i), real=True)
            i = i + 1
        return remapping

    def _mk_Comp(self, terms, lhs):
        pfx = str(next(self._tmp_var)) if lhs is None else str(lhs)
        accum = Symbol(pfx + "a", real=True)
        carry = Symbol(pfx + "c", real=True)
        tempv = Symbol(pfx + "t", real=True)
        na = self._CompAdd(terms, accum, carry, tempv)
        self._all_accum[accum] = na
        self._all_carry[carry] = na
        self._all_tempv[tempv] = na
        return na

    @classmethod
    def _is_Comp(cls, x):
        return isinstance(x, cls._CompAdd)

    def _single_pass(self, statements, pass_):
        new_stmts = []
        for st in statements:
            if isinstance(st, AssignmentBase):
                new_rhs = pass_(st.lhs, st.rhs, statements=new_stmts)
                if st.lhs not in self.created:
                    new_stmts.append(st.__class__(st.lhs, new_rhs))
            elif hasattr(st, "body"):
                assert isinstance(st.body, CodeBlock)
                new_body = CodeBlock(*self._single_pass(st.body.args, pass_))
                new_args = (
                    new_body if attr == "body" else getattr(st, attr)
                    for attr in st.__slots__
                )
                new_stmts.append(st.__class__(*new_args))
            else:
                new_stmts.append(st)  # no-op (e.g. BreakToken instance)
        return new_stmts

    def _pipeline(self):
        statements = [Assignment(*lr) for lr in self.repl]
        final_exprs = self.red
        for pass_ in self.passes:
            statements = self._single_pass(statements, pass_)
            final_exprs = [pass_(None, e, statements=statements) for e in final_exprs]
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
                score = self._analysis.get(lhs, 0) + reduce(
                    add, [self._analysis.get(k, 1) for k in _add.args]
                )
                if score >= self.limit or any(self._is_Comp(arg) for arg in _add.args):
                    na = self._mk_Comp(_add.args, lhs)
                    if _add is rhs and lhs is not None:
                        key = lhs

                    else:
                        key = next(self._csum_var)
                    self.created[key] = na
                    new_rhs = new_rhs.xreplace({_add: key})
                    break
            else:
                return new_rhs

        assert False

    def _pass_50_to_stmnts(self, lhs, rhs, *, statements):
        for csum in map(self.created.get, postorder_traversal(rhs)):
            if csum is None:
                continue
            self._pass_50_to_stmnts(lhs, csum.terms, statements=statements)
            if csum not in self.expanded:
                statements.extend(
                    csum.to_statements(
                        self.created,
                        self.expanded,
                        transients=self._trs_var,
                        **self.kw_to_statements
                    )
                )
        return rhs

    def _pass_60_xrepl(self, lhs, rhs, *, statements):
        return rhs.xreplace(self.created)

    def _group(self, x):
        all_accum, all_carry, rest = [], [], []
        for term in x.args:
            if term in self.created:
                all_accum.append(self.created[term].accum)
                all_carry.append(self.created[term].carry)
            elif term in self._all_accum:
                all_accum.append(term)
            elif term in self._all_carry:
                all_carry.append(term)
            else:
                rest.append(term)
        result = []
        if all_accum:
            result.append(ordered_add(*all_accum))
        if all_carry:
            result.append(ordered_add(*all_carry))
        if rest:
            result += rest
        result = reduce(add, result)
        return result

    def _has(self, term):
        return (
            term in self.created or term in self._all_accum or term in self._all_carry
        )

    def _pass_95_group(self, lhs, rhs, *, statements):
        new_rhs = rhs.replace(
            lambda s: s.is_Add and any(self._has(t) for t in s.args), self._group
        )
        return new_rhs

    def _pass_90_fin(self, lhs, rhs, *, statements):
        return rhs.replace(lambda x: self._is_Comp(x), lambda x: x.finalize())


class NeumaierTransformer(_CompensationTransformer):
    _CompAdd = _NeumaierAdd

    def __init__(self, *args, do_swap=False, **kwargs):
        #kw_to_statements = kwargs.pop(kw_to_statements, {})
        # ...
        super().__init__(*args, kw_to_statements=dict(do_swap=do_swap), **kwargs)

class TwoSumTransformer(_CompensationTransformer):
    _CompAdd = _TwoSumAdd

class FastTwoSumTransformer(_CompensationTransformer):
    _CompAdd = _FastTwoSumAdd


compensated_transformers = {
    'nocomp': NullTransformer,
    'kbn': NeumaierTransformer,
    '2sum': TwoSumTransformer,
    'fast2sum': FastTwoSumTransformer,
}
