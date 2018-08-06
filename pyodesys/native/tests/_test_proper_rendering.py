
def _test_render_native_code_cse(NativeSys):
    # regression test taken from chempy
    from pyodesys.symbolic import SymbolicSys
    from sympy import symbols, log, exp

    symbs = symbols(
        'N U A L NL t T He_dis Se_dis Cp_dis Tref_dis '
        'He_u Se_u Cp_u Tref_u Ha_agg Sa_agg Ha_as Sa_as Ha_f Sa_f '
        'R h k_B'
    )
    di = {s.name: s for s in symbs}

    class NS:
        pass

    ns = NS()
    ns.__dict__.update(di)

    def _gibbs(H, S, Cp, Tref):
        H2 = H + Cp*(ns.T - Tref)
        S2 = S + Cp*log(ns.T/Tref)
        return exp(-(H2 - ns.T*S2)/(ns.R*ns.T))

    def _eyring(H, S):
        return ns.k_B/ns.h*ns.T*exp(-(H - ns.T*S)/(ns.R*ns.T))

    k_agg = _eyring(di['Ha_agg'], di['Sa_agg'])
    k_as = _eyring(di['Ha_as'], di['Sa_as'])
    k_f = _eyring(di['Ha_f'], di['Sa_f'])
    k_dis = k_as*_gibbs(*[di[k] for k in ('He_dis', 'Se_dis', 'Cp_dis', 'Tref_dis')])
    k_u = k_f*_gibbs(*[di[k] for k in ('He_u', 'Se_u', 'Cp_u', 'Tref_u')])
    r_agg = k_agg*ns.U
    r_as = k_as*ns.N*ns.L
    r_f = k_f*ns.U
    r_dis = k_dis*ns.NL
    r_u = k_u*ns.N
    exprs = [
        -r_as + r_f + r_dis - r_u,
        -r_agg - r_f + r_u,
        r_agg,
        r_dis - r_as,
        r_as - r_dis
    ]
    odesys = SymbolicSys(zip(symbs[:5], exprs), symbs[5], params=symbs[6:])
    NativeSys.from_other(odesys)
