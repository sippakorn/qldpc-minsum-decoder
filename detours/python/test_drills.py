"""Tests for detours/python/drills.py (Detour: Python for QEC, Part I).

Run from the repo root, one set at a time:

    pytest detours/python -q -k p1      # then p2, ..., p7

The project's own `pytest -q` only collects tests/ (see pyproject.toml), so
these tests never mix with the research test suite.
"""

import itertools
import math
from collections import Counter

import numpy as np
import pytest

from detours.python.drills import (
    encode_all,
    error_batches,
    failure_counts,
    gf2_kernel,
    gf2_rref,
    in_rowspace,
    loglog_slope,
    make_hamming_decoder,
    min_distance,
    parity_check_from_columns,
    syndrome_histogram,
    weight_distribution,
)
from msd.toric import (
    cycle_code,
    gf2_rank,
    hypergraph_product,
    repetition_code,
    support_to_vector,
    syndrome_key,
    syndrome_of,
)

# Hamming [7, 4]: column j is the binary expansion of j + 1, bit 0 in row 0.
H7 = np.array(
    [[1, 0, 1, 0, 1, 0, 1],
     [0, 1, 1, 0, 0, 1, 1],
     [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)

# A generator matrix for it (worked example W2.1).
G7 = np.zeros((4, 7), dtype=np.uint8)
for _i, _sup in enumerate([(0, 1, 2), (0, 3, 4), (1, 3, 5), (0, 1, 3, 6)]):
    G7[_i, list(_sup)] = 1


def random_binary(rng, m, n, density=0.4):
    return (rng.random((m, n)) < density).astype(np.uint8)


def repetition_decoder(n):
    """The decoder of worked example W6.1 (minimum weight, odd n)."""
    def decode(s):
        e0 = np.concatenate([[0], np.cumsum(s) % 2]).astype(np.uint8)
        e1 = e0 ^ 1
        return e0 if e0.sum() <= e1.sum() else e1
    return decode


# ---------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------


def test_p1_hamming_7_4():
    H = parity_check_from_columns(range(1, 8), 3)
    assert H.dtype == np.uint8
    assert H.shape == (3, 7)
    assert np.array_equal(H, H7)


def test_p1_hamming_15_11():
    H = parity_check_from_columns(range(1, 16), 4)
    assert H.shape == (4, 15)
    assert np.array_equal(H[:, 8], [1, 0, 0, 1])  # column 8 holds 9 = 1001
    assert len({tuple(c) for c in H.T}) == 15 and H.any(axis=0).all()
    assert gf2_rank(H) == 4


def test_p1_any_iterable_and_padding():
    for cols in ([5, 2], (5, 2), np.array([5, 2])):
        assert np.array_equal(parity_check_from_columns(cols, 4), [[1, 0], [0, 1], [1, 0], [0, 0]])


# ---------------------------------------------------------------------------
# P2
# ---------------------------------------------------------------------------


def test_p2_encode_all_hamming():
    C = encode_all(G7)
    assert C.dtype == np.uint8 and C.shape == (16, 7)
    assert not C[0].any()
    assert np.array_equal(C[1], G7[0])
    assert np.array_equal(C[3], G7[0] ^ G7[1])  # rows 0 and 1 overlap: XOR, not OR
    assert not (C.astype(int) @ H7.T % 2).any()
    assert len(np.unique(C, axis=0)) == 16


def test_p2_encode_all_message_order():
    C = encode_all(np.eye(5, dtype=np.uint8))
    for i in range(32):
        assert np.array_equal(C[i], [(i >> j) & 1 for j in range(5)])


def test_p2_weight_distribution():
    A = weight_distribution(G7)
    assert np.array_equal(A, [1, 0, 0, 7, 7, 0, 0, 1])
    assert np.array_equal(weight_distribution(np.ones((1, 5), dtype=np.uint8)), [1, 0, 0, 0, 0, 1])
    G8 = np.hstack([G7, G7.sum(axis=1, keepdims=True) % 2])  # extended Hamming [8, 4]
    assert np.array_equal(weight_distribution(G8), [1, 0, 0, 0, 14, 0, 0, 0, 1])


# ---------------------------------------------------------------------------
# P3
# ---------------------------------------------------------------------------


def test_p3_in_rowspace_hamming():
    for msg in itertools.product((0, 1), repeat=4):
        c = np.array(msg, dtype=np.int64) @ G7 % 2
        assert in_rowspace(G7, c)
    for j in range(7):
        assert not in_rowspace(G7, support_to_vector(7, (j,)))
    assert in_rowspace(H7, H7[0] ^ H7[2])


def test_p3_rref_hamming():
    R, piv = gf2_rref(H7)
    assert list(piv) == [0, 1, 3]
    assert R.shape == (3, 7) and R.dtype == np.uint8
    R2, piv2 = gf2_rref(np.vstack([H7, H7[0] ^ H7[1]]))
    assert R2.shape == (3, 7) and list(piv2) == [0, 1, 3]


@pytest.mark.parametrize("seed", range(6))
def test_p3_rref_properties(seed):
    rng = np.random.default_rng(seed)
    M = random_binary(rng, 6, 9)
    R, piv = gf2_rref(M)
    r = gf2_rank(M)
    assert R.shape == (r, 9) and len(piv) == r
    assert list(piv) == sorted(piv)
    assert np.array_equal(R[:, piv], np.eye(r, dtype=np.uint8))
    for i, p in enumerate(piv):
        assert not R[i, :p].any()
    assert gf2_rank(np.vstack([R, M])) == r  # same row space


def test_p3_kernel_hamming_and_toric():
    K = gf2_kernel(H7)
    assert K.shape == (4, 7) and K.dtype == np.uint8
    assert not (H7.astype(int) @ K.T % 2).any() and gf2_rank(K) == 4
    _, HZ = hypergraph_product(cycle_code(3), cycle_code(3))
    K3 = gf2_kernel(HZ)
    assert K3.shape == (18 - gf2_rank(HZ), 18) == (10, 18)
    assert not (HZ.astype(int) @ K3.T % 2).any() and gf2_rank(K3) == 10


@pytest.mark.parametrize("seed", range(6))
def test_p3_kernel_random(seed):
    rng = np.random.default_rng(100 + seed)
    M = random_binary(rng, 5, 10)
    K = gf2_kernel(M)
    assert K.shape == (10 - gf2_rank(M), 10)
    assert not (M.astype(int) @ K.T % 2).any() and gf2_rank(K) == K.shape[0]


# ---------------------------------------------------------------------------
# P4
# ---------------------------------------------------------------------------


def test_p4_hamming_is_perfect():
    h1 = syndrome_histogram(H7, 1)
    assert isinstance(h1, Counter)
    assert len(h1) == 8 and set(h1.values()) == {1}
    assert h1[syndrome_key(np.zeros(3, dtype=np.uint8))] == 1


def test_p4_hamming_weight_two():
    h2 = syndrome_histogram(H7, 2)
    assert sum(h2.values()) == 1 + 7 + 21
    zero = syndrome_key(np.zeros(3, dtype=np.uint8))
    assert h2[zero] == 1
    assert all(c == 4 for k, c in h2.items() if k != zero)


def test_p4_repetition_5_is_perfect():
    h = syndrome_histogram(repetition_code(5), 2)
    assert len(h) == 16 and set(h.values()) == {1}


# ---------------------------------------------------------------------------
# P5
# ---------------------------------------------------------------------------


def test_p5_batches_cover_everything_in_order():
    chunks = list(error_batches(6, 2, 4))
    assert [len(s) for s, _ in chunks] == [4, 4, 4, 3]
    sups = np.vstack([s for s, _ in chunks])
    assert [tuple(r) for r in sups] == list(itertools.combinations(range(6), 2))
    for s, E in chunks:
        assert E.dtype == np.uint8 and E.shape == (len(s), 6)
        for row, sup in zip(E, s):
            assert np.array_equal(np.flatnonzero(row), sup)


def test_p5_batches_weight_zero():
    chunks = list(error_batches(5, 0, 10))
    assert len(chunks) == 1
    s, E = chunks[0]
    assert s.shape == (1, 0) and E.shape == (1, 5) and not E.any()


def test_p5_batches_are_lazy():
    # C(200, 4) = 64,684,950 supports: building them all would take minutes and GBs.
    s, E = next(error_batches(200, 4, 10))
    assert s.shape == (10, 4) and E.shape == (10, 200)


def test_p5_min_distance_classical():
    assert min_distance(H7, 5) == 3
    assert min_distance(parity_check_from_columns_ref(range(1, 16), 4), 4) == 3
    assert min_distance(repetition_code(5), 5) == 5
    assert min_distance(repetition_code(5), 4) is None


@pytest.mark.parametrize("d, expected", [(3, 3), (4, 4), (5, 4)])
def test_p5_min_distance_of_ker_HZ(d, expected):
    """The classical code ker H_Z of T_d has distance min(4, d) (see notes 5)."""
    _, HZ = hypergraph_product(cycle_code(d), cycle_code(d))
    assert min_distance(HZ, 4) == expected


def parity_check_from_columns_ref(cols, m):
    """Reference copy so P5 does not depend on your P1."""
    c = np.asarray(list(cols))
    return ((c[None, :] >> np.arange(m)[:, None]) & 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# P6
# ---------------------------------------------------------------------------


def test_p6_hamming_decoder_corrects_weight_one():
    decode = make_hamming_decoder(H7)
    assert callable(decode)
    assert np.array_equal(decode(np.zeros(3, dtype=np.uint8)), np.zeros(7))
    for j in range(7):
        e = support_to_vector(7, (j,))
        assert np.array_equal(decode(syndrome_of(H7, e)), e)


def test_p6_hamming_decoder_gives_up():
    H6 = H7[:, :6]  # shortened code: syndrome 111 is no column
    decode = make_hamming_decoder(H6)
    assert decode(np.array([1, 1, 1], dtype=np.uint8)) is None


def test_p6_failure_counts():
    assert failure_counts(make_hamming_decoder(H7), H7, 3) == [0, 0, 21, 35]
    assert failure_counts(repetition_decoder(5), repetition_code(5), 5) == [0, 0, 0, 10, 5, 1]
    give_up = lambda s: None
    assert failure_counts(give_up, H7, 3) == [math.comb(7, w) for w in range(4)]


# ---------------------------------------------------------------------------
# P7
# ---------------------------------------------------------------------------


def test_p7_slope_of_power_laws():
    ps = np.logspace(-5, -2, 30)
    assert loglog_slope(ps, 3 * ps**2) == pytest.approx(2.0, abs=1e-9)
    assert loglog_slope(ps, 10 * ps**3) == pytest.approx(3.0, abs=1e-9)


def test_p7_slope_of_hamming_block_failure():
    ps = np.logspace(-5, -4, 20)
    P = 1 - (1 - ps) ** 7 - 7 * ps * (1 - ps) ** 6
    assert loglog_slope(ps, P) == pytest.approx(2.0, abs=0.01)
