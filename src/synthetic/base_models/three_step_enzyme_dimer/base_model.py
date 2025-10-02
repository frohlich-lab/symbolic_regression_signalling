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

S = Monomer('S', sites=['k'])  # Substrate
P = Monomer('P', sites=['k'])  # Product
K = Monomer('K', sites=['p', 'd'])

Initial(K(p=None, d=None), Parameter('K0'))
Initial(S(k=None), Parameter('uP0'))
Initial(P(k=None), Parameter('pP0'))

kon_dim = Parameter('kon_dim')
koff_dim = Parameter('koff_dim')

# --- CHANGED: correct kon formula: kon = koff / Kd ---
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
    S(k=None) + K(p=None) |
    S(k=1) % K(p=1),
    kon_substrate,
    koff_substrate,
)

# ES on the dimer (same definition used in dynamic)
es = S(k=1) % K(p=1, d=2) % K(d=2)

# kcat = Parameter('kcat')

# # --- REMOVED: real catalytic conversion (irreversible sink) ---
# Rule('step2', es >> P(phospho='u', k=1) % K(p=1, d=2) % K(d=2), kcat)

# Product binding (adapted: remove 'phospho' site, match Monomer definitions)
Rule(
    'step3',
    P(k=1) % K(p=1) | P(k=None) + K(p=None),
    koff_product,
    kon_product,
)

# --- CHANGED: define Observable first, then capacity = kcat * [ES] ---
Expression(
    'kcat_cg',
    Observable('es_o', es),
)
