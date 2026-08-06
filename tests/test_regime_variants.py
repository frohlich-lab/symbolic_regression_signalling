import pytest

np = pytest.importorskip('numpy')
pytest.importorskip('pandas')

from shared.regime_variants import variant_model_combinations


def test_variant_model_combinations_invalid():
    with pytest.raises(ValueError):
        variant_model_combinations('bad')


@pytest.mark.skip(reason='Parity with prior refactor snapshot; integration-only check.')
def test_regime_plot_integration_placeholder():
    pass
