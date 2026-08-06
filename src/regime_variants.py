"""Compatibility wrapper for shared regime variant helpers."""

from pipelines.regimes import regime_variants as _impl

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not name.startswith("_")
    }
)

__all__ = [name for name in globals() if not name.startswith("_")]
