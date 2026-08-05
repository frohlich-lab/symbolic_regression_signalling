"""Generate the PySR hyperparameter grid for the OOD arm search.

Design notes (these belong in the methods section):
  * Only standard search/complexity hyperparameters and operator sets are varied.
    The custom-loss stability penalties stay at their defaults (1000/1000) in every
    arm -- they encode the dynamical-admissibility prior (d pERK/dt decreasing in
    pERK), so disabling them to gain R2 would be tuning away the prior.
  * maxsize starts at 18: the earlier ladder showed <=15 cannot express
    peak-and-decay (0 passing fits, max traj R2 ~0.57).
  * The unit of selection is the ARM, scored on aggregate over all its
    (marker, seed) fits -- not the best single fit within an arm.
  * Sample weighting is excluded: the custom Julia loss ignores dataset.weights,
    so those flags are a no-op.
"""
import itertools
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else "arms_grid.tsv"

MAX_ITER = [700, 1400]
POPULATIONS = [30, 60]
POP_SIZE = [30, 60]
MAX_SIZE = [18, 22, 26]
PARSIMONY = [0.1, 0.8, 3.0]
UNARY = [("none", ""), ("log", "--unary-operators log")]
BINARY = [("div", ""), ("nodiv", "--binary-operators + - *")]

TEST_SIZE = "0.2"

rows = []
for it, pops, ps, ms, par, (uname, uflag), (bname, bflag) in itertools.product(
    MAX_ITER, POPULATIONS, POP_SIZE, MAX_SIZE, PARSIMONY, UNARY, BINARY
):
    name = f"it{it}_p{pops}_ps{ps}_ms{ms}_par{str(par).replace('.', '')}_u{uname}_b{bname}"
    flags = [
        f"--max-iterations {it}",
        f"--populations {pops}",
        f"--population-size {ps}",
        f"--max-size {ms}",
        f"--parsimony {par}",
    ]
    if uflag:
        flags.append(uflag)
    if bflag:
        flags.append(bflag)
    rows.append((name, TEST_SIZE, " ".join(flags)))

with open(OUT, "w") as fh:
    fh.write("arm\ttest_size\textra_flags\n")
    for name, ts, flags in rows:
        fh.write(f"{name}\t{ts}\t{flags}\n")

print(f"{len(rows)} arms -> {OUT}")
print("grid:", len(MAX_ITER), "x", len(POPULATIONS), "x", len(POP_SIZE), "x",
      len(MAX_SIZE), "x", len(PARSIMONY), "x", len(UNARY), "x", len(BINARY))
for r in rows[:3]:
    print("  e.g.", r[0], "|", r[2])
assert len({r[0] for r in rows}) == len(rows), "duplicate arm names"
