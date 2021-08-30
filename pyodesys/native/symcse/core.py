from collections import defaultdict
from sympy import Symbol
from sympy.codegen.ast import Assignment, AssignmentBase, float64, value_const, Variable


class NullTransformer:
    """Perform no transformation."""

    def __init__(self, repl, red, parent=None, ignore=None):
        """From CSE result."""
        self.statements = [Assignment(lhs, rhs) for lhs, rhs in repl]
        self.final_exprs = red

    def statements_with_declarations(self, pred=lambda s: '[' not in s.name,
                                     type_=float64):
        """Conditionally add declarations."""
        return self._declare(self.statements, pred=pred, type_=type_)

    @staticmethod
    def _is_const(ctx_is_const, statements):
        for st in statements:  # filter(lambda s: isinstance(s, AssignmentBase), stmts):
            if isinstance(st, AssignmentBase):
                ctx_is_const[st.lhs] += 1 if type(st) is Assignment else 2
            elif hasattr(st, "body"):
                NullTransformer._is_const(ctx_is_const, st.body.args)
            else:
                pass  # no-op

    @staticmethod
    def _as_decl(ctx_seen, statements, is_const, *, pred, type_, lift=False):
        new_stmnts, lifted = [], []
        for st in statements:
            if isinstance(st, Assignment) and st.lhs not in ctx_seen and pred(st.lhs):
                ctx_seen.add(st.lhs)
                var = Variable(st.lhs, type=type_, attrs=[value_const] if
                               is_const[st.lhs] == 1 else [])
                if lift:
                    lifted.append(var.as_Declaration())
                else:
                    st = var.as_Declaration(value=st.rhs)
            elif hasattr(st, "body"):
                body_stmnts, body_lifted = NullTransformer._as_decl(
                    ctx_seen, st.body.args, is_const,
                    pred=pred, type_=type_, lift=True)
                if lift:
                    lifted.extend(body_lifted)
                else:
                    new_stmnts.extend(body_lifted)
            else:
                pass  # no-op, e.g. BreakToken()
            new_stmnts.append(st)
        return new_stmnts, lifted

    def _declare(self, stmts, *, pred=lambda x: True, type_=float64):
        seen = set()
        is_const = defaultdict(int)
        self._is_const(is_const, stmts)
        new_stmnts, lifted = self._as_decl(seen, stmts, is_const, pred=pred, lift=False, type_=type_)
        assert(len(lifted) == 0)
        return new_stmnts

    def apply_remapping(self, remapping):
        """Replace in statements & final_exprs."""
        self.statements = [st.xreplace(remapping) for st in self.statements]
        self.final_exprs = [e.xreplace(remapping) for e in self.final_exprs]

    def remapping_for_arrayification(self, template="m_glob[{0}]"):
        """Create a replacement dictionary."""
        remapping = {}
        i = 0
        for st in self.statements:
            if st.lhs in remapping:
                continue
            remapping[st.lhs] = Symbol(template.format(i), real=True)
            i = i + 1
        return remapping
