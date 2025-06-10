"""
A two-step enzyme model.

This implements a simple two-step enzyme model with a reversible binding
step and an irreversible catalytic step. The model is altered to not execute
the catalytic step, but to instead calculate the catalytic rate from the
concentration of the enzyme-substrate complex.
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

Initial(K(p=None), Parameter('K0'))
Initial(P(phospho='u', k=None), Parameter('uP0'))
Initial(P(phospho='p', k=None), Parameter('pP0'))

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
Rule('step2', es >> P(phospho='u', k=None) + K(p=None), kcat)

Expression(
    'kcat_cg',
    Observable('es_o', es),
)
