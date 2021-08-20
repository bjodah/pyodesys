from collections import defaultdict
from sympy import Symbol
from sympy.codegen.ast import Assignment, float64, value_const, Variable


class NullTransformer:
    """Perform no transformation."""

    def __init__(self, repl, red, parent=None, ignore=None):
        self.statements = [Assignment(lhs, rhs) for lhs, rhs in repl]
        self.final_exprs = red

    def statements_with_declarations(self, pred=lambda s: '[' not in s.name,
                                     type_=float64):
        return self._declare(self.statements, pred=pred, type_=type_)

    def _declare(self, stmts, *, pred=lambda x: True, type_=float64):
        seen = set()
        result = []

        is_const = defaultdict(int)
        for st in stmts:
            is_const[st.lhs] += 1 if type(st) is Assignment else 2
        for st in stmts:
            if isinstance(st, Assignment) and st.lhs not in seen and pred(st.lhs):
                seen.add(st.lhs)
                st = Variable(st.lhs, type=type_, attrs=[value_const] if
                              is_const[st.lhs] == 1 else []).as_Declaration(value=st.rhs)
            result.append(st)
        return result


    def apply_remapping(self, remapping):
        self.statements = [st.xreplace(remapping) for st in self.statements]
        self.final_exprs = [e.xreplace(remapping) for e in self.final_exprs]

    def remapping_for_arrayification(self, template="m_glob[{0}]"):
        remapping = {}
        i = 0
        for st in self.statements:
            if st.lhs in remapping:
                continue
            remapping[st.lhs] = Symbol(template.format(i), real=True)
            i = i + 1
        return remapping
