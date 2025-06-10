# constants.py

# Found with parameter inference script train.py
# Extracted from petab/average_optimized_parameters.txt

# Parameters in linear scale (Best parameters)
BIND_MTX_EA = 1.2969326291682852   # Linear scale
BIND_MTX_DG = 6.257184358190384    # Linear scale
BIND_MTX_PHI = 0.0   # Linear scale

BIND_S_EA = -3.096740439352854     # Linear scale
BIND_S_DG = 9.484662689867596     # Linear scale
BIND_S_H_DDG = 1.5773146537418168  # Linear scale
BIND_S_PHI = 0.0    # Linear scale

# Parameters in log10 scale (converted to linear values)
K_CAT_OFF = 10**(1.4620980976568567)   # Log10 scale -> Linear value
K_CAT_ON = 10**(3.842626908576797)   # Log10 scale -> Linear value
K_ON = 10**(-0.019371247966819088)    # Log10 scale -> Linear value
SCALE_PRODUCT_C0 = 10**(-4.794748480502125)  # Log10 scale -> Linear value

# Notes:
# - Parameters with log10 scales have been converted to their linear values.
# - Ensure to use these linear values directly in computations.

# Initial values
INITIAL_ACTIVATOR_CONC = 1000
ENZYME_CONC = 10
SUBSTRATE_CONC = 50000

LATER_ACTIVATORS_CONC = 0.0

FINAL_TIME_EXPERIMENT = 44  # minutes
