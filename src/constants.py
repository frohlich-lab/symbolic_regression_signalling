"""Compatibility wrapper for shared constants.

The implementation lives in `src/shared/constants.py`. This module existed as a
byte-identical second copy, which meant every change had to be made twice -- and
whichever copy was forgotten would silently diverge. It is now a re-export, matching
the pattern `src/regime_variants.py` already uses.

It has to stay: six live modules import it by the bare name, resolved through the
`sys.path.insert(0, SRC_ROOT)` at the top of each pipeline script --
`src/pipelines/regimes/{kinetic,noise,mm_deviation,dataset_size,timepoint}_regimes.py`
and `src/sr_models/pysr_model.py` all do `from constants import ...`. Deleting this
file would break all six; rewriting their imports is a separate change.
"""

from shared import constants as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("_")
    }
)

__all__ = [name for name in globals() if not name.startswith("_")]
