"""Manuscript naming convention for figure and table labels.

Every join in the pipeline is keyed on the marker names as they appear in the
source data (Lun et al., Mol Cell 2019): ``PIP5K3`` for the construct,
``p-RAF``/``p-PDK1`` for the readouts. Those keys are load-bearing -- the raw
CSV columns, the per-shard run directories and the deposited artefacts all use
them -- so nothing here renames data. This is the display layer, applied on the
way into a panel or a typeset table.

The convention, applied everywhere:

* An overexpression **context** is a transfected gene, so it carries its
  HGNC-approved symbol. Exactly one context in the source data does not:
  ``PIP5K3`` is a previous symbol for ``PIKFYVE`` (HGNC:23785).
* A **readout** is a phospho-antibody, not a gene. Four of the seven are
  pan-isoform -- one antibody, two genes -- so a single HGNC symbol would
  misdescribe the reagent. Readouts therefore keep their protein common names,
  and :data:`READOUT_GENES` carries the gene/site/clone mapping that Table S11
  typesets. This is also why the same node must not be drawn as ``p90RSK`` in
  one figure and ``RPS6KA1`` in another: under this convention it is ``p90RSK``.
"""
from __future__ import annotations

import re

import pandas as pd

# Contexts whose source-data name is not the HGNC-approved symbol. Verified
# against the HGNC REST API; PIP5K3 is the only one of the 40.
GENE_DISPLAY = {
    "PIP5K3": "PIKFYVE",
}

# Plasmid disambiguator carried by two contexts, e.g. "DUSP10 (P2)". It is not
# part of the symbol, so it survives the mapping untouched.
_SUFFIX = re.compile(r"^(?P<symbol>[^\s(]+)(?P<suffix>.*)$")

# Readout -> (HGNC symbol(s), phospho-site, clone). Antibodies as recorded in
# the Key Resources Table of Lun et al., Mol Cell 2019, the source of the
# measurements. A "/" in the symbol column means the antibody is pan-isoform
# and detects both gene products; it is not a choice between them.
READOUT_GENES = [
    ("p-ERK1/2",   "MAPK3 / MAPK1",   "Thr202/Tyr204",          "20A"),
    ("p-MEK1/2",   "MAP2K1 / MAP2K2", "Ser221",                 "166F8"),
    ("p-RAF",      "RAF1",            "Ser259",                 "polyclonal"),
    ("p-p90RSK",   "RPS6KA1",         "Ser380",                 "D5D8"),
    ("p-MKK3/6",   "MAP2K3 / MAP2K6", "Ser189 (MKK3) / Ser207 (MKK6)", "D8E9"),
    ("p-MAPKAPK2", "MAPKAPK2",        "Thr334",                 "27B7"),
    ("p-PDK1",     "PDPK1",           "Ser241",                 "J66-653.44.22"),
]

# Pathway nodes drawn in the Fig. 4 and Fig. 5 schematics but never measured,
# recorded so the caption can say which gene products they stand for. RAS and
# PI3K are families rather than single genes, which is why they stay as drawn.
SCHEMATIC_NODES = {
    "EGFR": "EGFR",
    "RAS": "HRAS / KRAS / NRAS",
    "PI3K": "PIK3CA / PIK3R1",
}

# D5D8 is raised against the RSK1 site and cross-reacts with the homologous
# site on RSK2/3, so the readout is not RPS6KA1-specific. Table S11 reports the
# nominal target; this note is the caveat that belongs with it.
RSK_CROSSREACTIVITY = (
    "The p-p90RSK clone D5D8 is raised against the RPS6KA1 Ser380 site and "
    "cross-reacts with the homologous site on RPS6KA2 and RPS6KA3."
)


def gene_label(name: str) -> str:
    """Source-data context name -> the symbol used in figures and tables.

    Controls (``untransfected1``, ``FLAG-GFP1``) are not genes and pass through
    unchanged, as does any context already on its approved symbol.

    >>> gene_label("PIP5K3")
    'PIKFYVE'
    >>> gene_label("DUSP10 (P2)")
    'DUSP10 (P2)'
    """
    m = _SUFFIX.match(str(name).strip())
    if m is None:
        return str(name)
    return GENE_DISPLAY.get(m["symbol"], m["symbol"]) + m["suffix"]


def gene_labels(names) -> pd.Series:
    """Vectorised :func:`gene_label`, for a column of context names."""
    return pd.Series(list(names)).map(gene_label)


def readout_table() -> pd.DataFrame:
    """Table S11: what each phospho-readout measures, in HGNC terms."""
    return pd.DataFrame(
        READOUT_GENES,
        columns=["Readout", "HGNC symbol", "Phospho-site", "Antibody clone"])
