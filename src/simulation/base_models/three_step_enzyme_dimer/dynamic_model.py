"""
A three-step dineric enzyme model.

This implements a simple three-step dimeric enzyme model with a reversible binding
step and an irreversible catalytic step.
"""

from pysb import (
    Expression,
    Initial,
    Model,
    Monomer,
    Observable,
    Parameter,
    Rule,
)

model = Model('three_step_enzyme_dimer')

P = Monomer('P', sites=['phospho', 'k'], site_states={'phospho': ['u', 'p']})

K = Monomer('K', sites=['p', 'd'])

K0 = Parameter('K0')
Initial(K(p=None, d=None), K0)
u0 = Parameter('uP0')
Initial(P(phospho='u', k=None), u0)
p0 = Parameter('pP0')
Initial(P(phospho='p', k=None), p0)

kon_dim = Parameter('kon_dim')
koff_dim = Parameter('koff_dim')

koff_substrate = Parameter('koff_substrate')
k_d_substrate = Parameter('kD_substrate')
kon_substrate = Expression('kon', koff_substrate * k_d_substrate)

koff_product = Parameter('koff_product')
kon_product = Expression('kon_product', koff_product * k_d_substrate)

Rule(
    'dimerisation',
    K(d=None) + K(d=None) | K(d=1) % K(d=1),
    kon_dim,
    koff_dim,
)

Rule(
    'step1',
    P(phospho='u', k=None) + K(p=None) | P(phospho='u', k=1) % K(p=1),
    kon_substrate,
    koff_substrate,
)

es = P(phospho='u', k=1) % K(p=1, d=2) % K(p=None, d=2)
kcat = Parameter('kcat')
Rule('step2', es >> P(phospho='p', k=1) % K(p=1, d=2) % K(p=None, d=2), kcat)

Rule(
    'step3',
    P(phospho='p', k=1) % K(p=1) | P(phospho='p', k=None) + K(p=None),
    koff_product,
    kon_product,
)

krev = Parameter('krev')
Rule('reverse', P(phospho='p') >> P(phospho='u'), krev)

kinact = Parameter('kinact')
kact = Expression('kact', kinact * K0)
Rule('activation', None >> K(p=None, d=None), kact)
Rule('inactivation', K() >> None, kinact, delete_molecules=True)

Observable('P_p', P(phospho='p'))
Observable('tK', K())

Expression(
    'kcat_cg',
    Observable('es_o', es),
)
