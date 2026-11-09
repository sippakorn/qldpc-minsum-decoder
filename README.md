# qldpc-minsum-decoder

Decoding radius of min-sum with decimation on quantum LDPC codes (Plan A, problem 4b; target ISIT 2027).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
pytest -q
```

## Layout

```
msd/toric.py         L1   codes, cosets, brute-force decoder
msd/ms.py            L2   integer syndrome min-sum (numba)
msd/trees.py         L3   decoding-tree solver, the second reference
msd/symmetry.py      L4   Aut(T_d), orbits, G_s, invariant solutions
msd/validate.py      L5   du Crest reproduction
msd/decimation.py    L6   MS-D_k with R_loc and R_guess
msd/enumerate.py     Gate 1 sprint
msd/hgp.py, planar.py  L8
tests/               one test file per module
notebooks/           NB1 to NB8
results/             CSV tables from the sprints
```

## Conventions (fixed in Lesson 1)

- T_d: d x d torus, qubits on the n = 2d^2 edges. Vertex v(x,y) and face f(x,y) have index x + d*y;
  horizontal edge h(x,y) has index x + d*y, vertical edge u(x,y) has index d^2 + x + d*y.
- X errors e in F_2^n; Z checks on vertices, s = H_Z e; X stabilizers on plaquettes (rows of H_X).
- A correction succeeds iff e + e_hat is in the row space of H_X.
- Min-sum: integer messages, flooding, value-based hard decision (LLR exactly 0 means "no error").
  Symmetry is broken only on purpose (clamps, splitter rows).

## Status

| Lesson | Module | State |
|---|---|---|
| L1 | `msd/toric.py`, `tests/test_toric.py`, `notebooks/NB1_toric.ipynb` | scaffold: 5 TODO functions |

## Authorship (IEEE AI policy)

Scaffolds, plumbing, plots and tests are written with Claude (Anthropic). Functions marked `TODO`
are written by Sippakorn Santavisith, as are proofs and paper text. The project study log records
which is which, lesson by lesson.
