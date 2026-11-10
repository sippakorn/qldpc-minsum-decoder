# Detour: Python for QEC

Study material, not research code. Nothing in `msd/` imports from here.

- Notes and exercises: the "Detour · Python for QEC" doc (Notes and Exercises tabs).
- `part1.ipynb`: Part I, the Python that NB1 uses. Worked examples W1.1 to W7.2, then one exercise cell per set.
- `drills.py`: the functions you write (sets P1 to P7; P3.3 and P5.1 are optional).
- `test_drills.py`: their tests. Run one set at a time from the repo root:

```bash
pytest detours/python -q -k p1    # ... up to p7
```

The project's own `pytest -q` collects only `tests/`, so these tests never mix with the research suite.
Parts II to IV (numba, graphs and permutations, Monte Carlo) are added when Lessons 2, 3 and 8 start.
