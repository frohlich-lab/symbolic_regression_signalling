"""
A two-step enzyme model.

This implements a simple two-step enzyme model with a reversible binding
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

model = Model('two_step_enzyme')

P = Monomer('P', sites=['phospho', 'k'], site_states={'phospho': ['u', 'p']})

K = Monomer('K', sites=['p'])

K0 = Parameter('K0')
Initial(K(p=None), K0)
u0 = Parameter('uP0')
Initial(P(phospho='u', k=None), u0)
p0 = Parameter('pP0')
Initial(P(phospho='p', k=None), p0)

koff_substrate = Parameter('koff_substrate')
k_d_substrate = Parameter('kD_substrate')
kon_substrate = Expression('kon', koff_substrate * k_d_substrate)

Rule(
    'step1',
    P(phospho='u', k=None) + K(p=None) | P(phospho='u', k=1) % K(p=1),
    kon_substrate,
    koff_substrate,
)

es = P(phospho='u', k=1) % K(p=1)
kcat = Parameter('kcat')
Rule('step2', es >> P(phospho='p', k=None) + K(p=None), kcat)

krev = Parameter('krev')
Rule('reverse', P(phospho='p') >> P(phospho='u'), krev)

kinact = Parameter('kinact')
kact = Expression('kact', kinact * K0)
Rule('activation', None >> K(p=None), kact)
Rule('inactivation', K() >> None, kinact, delete_molecules=True)

Observable('P_p', P(phospho='p'))
Observable('tK', K())

Expression(
    'kcat_cg',
    Observable('es_o', es),
)

d_p = Monomer('dP')
Rule(
    'delayedPon',
    P(phospho='p') >> d_p() + P(phospho='p'),
    krev,
)
Rule('delayedPoff', d_p() >> None, krev)
P_p_d = Observable('P_p_d', d_p())
Expression('P_u_d', p0 + u0 - P_p_d)

d_k = Monomer('dK')
Rule(
    'delayedKon',
    K() >> d_k() + K(),
    kinact,
)
Rule('delayedKoff', d_k() >> None, kinact)
Observable('K_d', d_k())
