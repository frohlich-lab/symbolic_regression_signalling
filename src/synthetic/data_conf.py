"""Cofiguration file for the data generation."""

PAR_LB = -5.0  # lower parameter boundary
PAR_UB = 3.0  # upper parameter boundary
N_SAMPLES = {
    'train': 2**16,
    'val': 2**14,
    'test': 2**14,
}  # number of samples for each dataset
N_T = 20  # number of time points
