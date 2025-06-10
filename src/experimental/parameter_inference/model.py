"""A three-step enzyme model for the MTX biosensor with cofactor binding and product release.

see https://doi.org/10.1038/s41467-021-27184-w
"""

from pysb import (
    ANY,
    EnergyPattern,
    Expression,
    Initial,
    Model,
    Monomer,
    Observable,
    Parameter,
    Rule,
)

from parameter_inference.data import MTX, PRODUCT_VARIABLE, SUBSTRATE, SWITCH

model = Model("MTX_biosensor")

glucose = Monomer("glucose", sites=["switch", "H"], site_states={"H": ["H0", "H1"]})
switch = Monomer(
    "switch", sites=["s", "mtx", "state"], site_states={"state": ["on", "off"]}
)
mtx = Monomer("mtx", sites=["switch"])

Initial(glucose(switch=None, H="H1"), Parameter(SUBSTRATE))
Initial(switch(s=None, mtx=None, state="off"), Parameter(SWITCH))
Initial(mtx(switch=None), Parameter(MTX))

# binding mtx
Ea = Parameter("bind_mtx_Ea", nonnegative=False)
dG = Parameter("bind_mtx_dG", nonnegative=False)
# increased stability upon binding -> destabilization via koff
phi = Parameter("bind_mtx_phi", 0.0)
Rule(
    "binding_mtx",
    mtx(switch=None) + switch(mtx=None) | mtx(switch=1) % switch(mtx=1),
    phi,
    Expression("Ea0_bind_mtx", -phi * dG - Ea),
    energy=True,
)
EnergyPattern("ep_bind_mtx", mtx(switch=1) % switch(mtx=1), dG)


# substrate binding
Ea = Parameter("bind_s_Ea", nonnegative=False)
dG = Parameter("bind_s_dG", nonnegative=False)
ddG = Parameter("bind_s_H_ddG", nonnegative=False)
# product release can be rate limiting -> destabilization via koff
phi = Parameter("bind_s_phi", 0.0)
Rule(
    "binding_s",
    switch(s=None) + glucose(switch=None) | switch(s=1) % glucose(switch=1),
    phi,
    Expression("Ea0_bind_s", -phi * dG - Ea),
    energy=True,
)
EnergyPattern("ep_bind_s", switch(s=1) % glucose(switch=1), dG)
EnergyPattern("ep_bind_s_H", switch(s=1) % glucose(switch=1, H="H0"), ddG)


# substrate conversion
Rule(
    "conversion_off",
    switch(s=1, mtx=None) % glucose(switch=1, H="H1")
    >> switch(s=1, mtx=None) % glucose(switch=1, H="H0"),
    Parameter("k_cat_off"),
)
Rule(
    "conversion_on",
    switch(s=1, mtx=ANY, state="on") % glucose(switch=1, H="H1")
    >> switch(s=1, mtx=ANY, state="on") % glucose(switch=1, H="H0"),
    Parameter("k_cat_on"),
)

# switch activation
Rule(
    "activation",
    switch(state="off") >> switch(state="on"),
    Parameter("k_on"),
)

obs = Observable(PRODUCT_VARIABLE, glucose(H="H0"))
obs_1 = Observable('substrate_unbound', glucose(H="H1"))
obs_2 = Observable('activated_switch', switch(state="on"))
obs_3 = Observable('product', glucose(H="H0"))
