"""
A three-step enzyme model.

This implements a simple three-step enzyme model with a reversible substrate binding
step, an irreversible catalytic step and a reversible product binding step. The model is altered to not execute
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

model = Model('three_step_enzyme')

S = Monomer('S', sites=['k'])
P = Monomer('P', sites=['k'])
K = Monomer('K', sites=['p'])

Initial(K(p=None), Parameter('K0'))
Initial(S(k=None), Parameter('uP0'))
Initial(P(k=None), Parameter('pP0', 3.0))

koff_substrate = Parameter('koff_substrate')
k_d_substrate = Parameter('kD_substrate')
kon_substrate = Expression('kon', koff_substrate * k_d_substrate)

koff_product = Parameter('koff_product')
kon_product = Expression('kon_product', koff_product * k_d_substrate)

Rule(
    'step1',
    S(k=None) + K(p=None) | S( k=1) % K(p=1),
    kon_substrate,
    koff_substrate,
)

es = S(k=1) % K(p=1)
# kcat = Parameter('kcat')
# Rule('step2', es >> P(phospho='u', k=1) % K(p=1), kcat)

Rule(
    'step3',
    P(k=1) % K(p=1) | P(k=None) + K(p=None),
    koff_product,
    kon_product,
)

Expression(
    'kcat_cg',
    Observable('es_o', es),
)
