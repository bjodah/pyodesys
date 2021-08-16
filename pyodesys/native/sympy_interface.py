#!/usr/bin/env python3
"""Use this utility for symbolic treatment.

Example usage:

$ python3 sympy_interface.py demo1 | clang-format --style=Google | batcat -pl C

"""
import pprint
from collections import defaultdict
from functools import reduce
from operator import add
from sympy import (
    Abs, Add, Basic, ccode, cse, exp, Expr, numbered_symbols, Piecewise, pi,
    postorder_traversal, preorder_traversal, pycode, Symbol, Tuple
)
from sympy.codegen import Assignment, aug_assign, CodeBlock
from sympy.codegen.ast import Token, Variable, float64
from sympy.printing.precedence import PRECEDENCE, precedence


class OrderedAdd(Expr):
    """Printed with parenthesis, useful for floating point math."""

    nargs = -1
    precedence = PRECEDENCE["Add"] - 1

    def _ccode(self, printer):
        return " + ".join(printer.parenthesize(arg, precedence(self)) for arg in self.args)

    _pythoncode = _ccode


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
            # print(f"/*{term, term in existing}*/")
            if term in existing: #.values():
                # neum.append(term)
                neum.append(existing[term])
            else:
                ordinary.append(term)
        # print(neum, ordinary, existing)
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


class _NeumaierTransformer:
    """Transform Add instances in CSEs to use compensated sum.

    Parameters
    ----------
    up_to: int, [0-100]
        Code is guaranteed to compile at levels 0 (no passes, no compensation)
        and 100 (all passes).
    """

    def __init__(self, repl, red, *, tmp_pfx="t", neu_pfx="n", up_to=100, limit=3):
        pprint.pprint(repl)
        pprint.pprint(red)
        print("")
        self.repl = repl
        self.red = red
        self.limit = limit

        self.created = {}
        self._all_accum = set()
        self._all_carry = set()
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
        # print("\ncreated: ", self.created)
        # print("")
        # print(self._analysis)
        # print(self.statements)

    def statements_and_expressions(self):
        return self._declare(self.statements), self.final_exprs

    def _mk_Neu(self, terms, lhs):
        pfx = str(next(self._tmp_var)) if lhs is None else str(lhs)
        accum = Symbol(pfx+'a')
        carry = Symbol(pfx+'c')
        tempv = Symbol(pfx+'t')
        self._all_accum.add(accum)
        self._all_carry.add(carry)
        return _NeumaierAdd(terms, accum, carry, tempv)

    @staticmethod
    def _is_Neu(x):
        return isinstance(x, _NeumaierAdd)

    def _declare(self, stmts, *, type_=float64):
        seen = set()
        result = []
        for st in stmts:
            if isinstance(st, Assignment) and st.lhs not in seen:
                seen.add(st.lhs)
                st = Variable(st.lhs, type=type_).as_Declaration(value=st.rhs)
            result.append(st)
        return result

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
        # print(f"/*{self._analysis}*/")
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
        #new_rhs = rhs.xreplace(self.created)
        new_rhs = rhs
        while True:
            for _add in filter(lambda x: x.is_Add, postorder_traversal(new_rhs)):
                score = self._analysis.get(lhs, 0) + reduce(add, [self._analysis.get(k, 1) for k in _add.args])
                if score >= self.limit or any(self._is_Neu(arg) for arg in _add.args):
                    na = self._mk_Neu(_add.args, lhs)
                    if _add is rhs and lhs is not None:
                        key = lhs

                    else:
                        key = next(self._neu_var)
                        #statements.append(Assignment(key, na))
                    self.created[key] = na
                    # print(new_rhs, _add)
                    new_rhs = new_rhs.xreplace({_add: key})
                    break
            else:
                return new_rhs

        assert False
        # if debug and lhs is None:
        #     print("new_rhs: ", new_rhs)
        #return
        # if debug and lhs is None:
        #     print("")
        #     print("_add: ", _add)
        #     print("")
        #     print("na: ", na)
        #     print("")
        #     print("new_rhs: ", new_rhs)

        #return new_rhs

    # def _pass_11_(self, lhs, rhs, *, statements):
    #     return self._pass_10_create_nodes(lhs, rhs, statements=statements, debug=True)

    def _pass_50_to_stmnts(self, lhs, rhs, *, statements):
        for arg in postorder_traversal(rhs):
            # print(arg, self._is_Neu(arg))
            if arg in self.created:
                for t in list(self.created[arg].terms)+[arg]:
                    if t in self.created:
                        if arg not in self.expanded:
                            statements.extend(self.created[t].to_statements(self.created, self.expanded))
                # elif self._is_Neu(arg):
                #     print(".")
                #     statements.extend(arg.to_statements(self.created, self.expanded))
        return rhs

    def _pass_60_xrepl(self, lhs, rhs, *, statements):
        return rhs.xreplace(self.created)

    def _group(self, x):
        # print(self.created)
        # print(x)###
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
        # print('//', all_accum, all_carry)
        if all_accum and all_carry:
            return OrderedAdd(OrderedAdd(*all_accum), OrderedAdd(*all_carry))
        else:
            return OrderedAdd(*(all_accum+all_carry))

    def _pass_95_group(self, lhs, rhs, *, statements):
        # Not sure if this pass is still needed
        return rhs.replace(lambda s: s.is_Add and any(
            t in self.created or t in self._all_accum or t in self._all_carry for t in s.args
        ), self._group)

    def _pass_90_fin(self, lhs, rhs, *, statements):
        return rhs.replace(lambda x: self._is_Neu(x), lambda x: x.finalize())


def _compensated_code(case, **kwargs):
    repl, red = cse(case.exprs)
    nm = _NeumaierTransformer(repl, red, **kwargs)
    statements, new_exprs = nm.statements_and_expressions()
    for i, new_expr in enumerate(new_exprs):
        statements.append(Assignment(Symbol("out[%d]" % i), new_expr))
    return CodeBlock(*statements)


class Case:
    """test case."""

    vals_ref = None

    @classmethod
    def check(cls):
        """Perform self-check."""
        return _NeumaierTransformer(*cse(cls.exprs))


class _Case0(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(3)]))
    a = reduce(add, x[:3])
    exprs = [
        a,
        a+1,
    ]


class _Case1(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(6)]))
    a = reduce(add, x[:3])
    b = reduce(add, x[3:6])
    exprs = [
        a,
        b,
        a + b,
        a + b + 1,
    ]


class _Case2(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(6)]))
    a = reduce(add, x[:3])
    b = reduce(add, x[3:6])
    exprs = [
        a,
        b,
        a + b + 1,
        a + b + 1 + pi,
    ]


class _Case3(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(10)]))
    a = reduce(add, x[:3])
    b = reduce(add, x[3:6])
    c = reduce(add, x[6:9])
    exprs = [
        a + 1,
        b + 2,
        c + 3,
        a + b + x[9],
        a + b + c + x[8]
    ]


class _Case4(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(3)]))
    a = 1*x[0]
    b = 2*x[1]
    c = 3*x[2]
    exprs = [
        a + b,
        a + b + c,
        a + b + c + 1
    ]


def real(name):
    return Symbol(name, real=True)


class _Case5(Case):
    x = list(map(real, ["y[%d]" % i for i in range(5)]))
    a = x[0]*x[1]**2 + x[2]**3 + x[3]**4
    b = 3*x[3] + 4*x[4]
    c = exp(1*x[1] + 2*x[2] + b)
    d = (1*x[1] + 2*x[2] + 3*x[3])/b
    exprs = [a, a/42, b+c, c*d, b+exp(x[0])]


class _Case6(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(2)]))
    a = reduce(add, x[:])
    exprs = [
        a,
        a+1,
    ]


class _Case7(Case):  # no Neumaier
    x = list(map(Symbol, ["y[%d]" % i for i in range(2)]))
    a = reduce(add, x[:])
    exprs = [
        a,
        a/3,
    ]

    @classmethod
    def check(cls):
        tr = super().check()
        assert(len(tr.statements) <= 1)


class _Case8(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(6)]))
    a = reduce(add, x[:3])
    b = reduce(add, x[3:])
    exprs = [
        a,
        b,
        a+b
    ]

class _Case9(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(6)]))
    a = reduce(add, x[:3])
    b = reduce(add, x[3:])
    exprs = [
        exp(a) + exp(b),
        a+b
    ]

class _Case10(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(6)]))
    a = reduce(add, x[:3])
    b = reduce(add, x[3:])
    exprs = [
        a/b,
        2*a + 3*b
    ]

class _Case11(Case):
    x = list(map(Symbol, ["y[%d]" % i for i in range(4)]))
    exprs = [sum(x)]
    vals_ref = ([1e14, 1.0, -1e14, 1.0], [2.0])


cases = [v for k, v in globals().items()
         if k.startswith("_Case") and Case in v.__mro__]

template_c = (
    "#include <math.h>\n"
    "void f(double * const out, const double * const y) {%s}"
)
template_py = "def f(out, y):\n    %s"

if __name__ == '__main__':

    def demo_c(index=0, up_to=100, limit=3):
        """Demo C-code."""
        case = cases[index]
        print('/* ' + str(case.exprs))
        code_block = _compensated_code(case, up_to=up_to, limit=limit)
        print('*/')
        # print(code_block)
        src = ccode(code_block)
        print(template_c % src)

    def demo_py(index=0, up_to=100, limit=3):
        """Demo Python-code."""
        src = pycode(_compensated_code(cases[index], up_to=up_to, limit=limit))
        print(template_py % src.replace('\n', '\n    '))

    import argh
    argh.dispatch_commands([demo_c, demo_py])
