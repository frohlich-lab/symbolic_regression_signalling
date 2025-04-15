"""
A three-step dimeric enzyme model.

This implements a three-step dimeric enzyme model with a reversible substrate binding step and a enzyme dimerisation
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

model = Model('three_step_enzyme_dimer')

uP = Monomer('uP', sites=['k'])
pP = Monomer('pP', sites=['k'])
K = Monomer('K', sites=['p', 'd'])

Initial(K(p=None, d=None), Parameter('K0'))
Initial(uP(k=None), Parameter('uP0'))
Initial(pP(k=None), Parameter('pP0'))

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
    uP(k=None) + K(p=None) | uP(k=1) % K(p=1),
    kon_substrate,
    koff_substrate,
)

es = uP(k=1) % K(p=1, d=2) % K(d=2)
kcat = Parameter('kcat')
Rule('step2', es >> uP(k=1) % K(p=1, d=2) % K(d=2), kcat)

Rule(
    'step3',
    pP(k=1) % K(p=1) | pP(k=None) + K(p=None),
    koff_product,
    kon_product,
)

Expression(
    'kcat_cg',
    Observable('es_o', es),
)
