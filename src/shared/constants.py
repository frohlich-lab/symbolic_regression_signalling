from typing import Optional

PYSR_CONFIG = {
    'niterations': 350,
    'population_size': 30,
    'populations': 15,
    'binary_operators': ["+", "*", "/", "-"],
    'unary_operators': ["exp", "log", "sqrt"],
    'maxsize': 20,
    'parsimony': 1,
    'verbosity': 0,
    'batching': True,
    'annealing': True,
    'elementwise_loss': "my_loss(x,y)=(log(max(x,1e-25))-log(max(y,1e-25)))^2",
    'random_state': 42,
}


def pysr_operator_config(variant: Optional[str]) -> dict:
    if not variant:
        return {}
    variant_norm = variant.strip().lower()
    if variant_norm == "sqssa":
        return {
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [],
        }
    if variant_norm == "tqssa":
        return {
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["sqrt", "square"],
            "maxsize": 45,
        }
    return {}
