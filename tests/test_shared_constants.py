from shared.constants import pysr_operator_config


def test_pysr_operator_config_known_variants():
    assert pysr_operator_config('sqssa')['unary_operators'] == []
    assert 'sqrt' in pysr_operator_config('tqssa')['unary_operators']
