"""Tests for msd/toric.py (NB1, Lesson 1).

Run:  pytest tests/test_toric.py -q
Each group tests one TODO, so you can work through them in order:
    pytest tests/test_toric.py -q -k checks      # TODO 1
    pytest tests/test_toric.py -q -k logicals    # TODO 2
    pytest tests/test_toric.py -q -k coset       # TODO 3
    pytest tests/test_toric.py -q -k table       # TODO 4
    pytest tests/test_toric.py -q -k radius      # TODO 5
"""

from collections import deque

import numpy as np
import pytest

from msd.toric import (
    coset_test,
    css_parameters,
    cycle_code,
    decoding_radius,
    edge_coords,
    face_index,
    gf2_rank,
    hedge_index,
    hypergraph_product,
    is_degenerate,
    make_table_decoder,
    min_weight_table,
    repetition_code,
    support_to_vector,
    syndrome_key,
    syndrome_of,
    toric_checks,
    toric_logicals,
    vector_to_support,
    vedge_index,
    vertex_index,
)

DS = [3, 4, 5, 6]


def plaquette_pair(d):
    """Plan A's plaquette error e1 + e2 = h(0,0) + u(1,0) and its twin e3 + e4."""
    e = (hedge_index(d, 0, 0), vedge_index(d, 1, 0))
    twin = (hedge_index(d, 0, 1), vedge_index(d, 0, 0))
    return tuple(sorted(e)), tuple(sorted(twin))


# ---------------------------------------------------------------------------
# Supplied plumbing (should pass before you write anything)
# ---------------------------------------------------------------------------


def test_plumbing_indexing_roundtrip():
    d = 5
    for x in range(d):
        for y in range(d):
            assert edge_coords(d, hedge_index(d, x, y)) == ("h", x, y)
            assert edge_coords(d, vedge_index(d, x, y)) == ("v", x, y)
    assert hedge_index(d, -1, 0) == hedge_index(d, d - 1, 0)
    assert vertex_index(d, d, d) == 0


def test_plumbing_gf2_rank():
    assert gf2_rank(np.eye(4, dtype=np.uint8)) == 4
    assert gf2_rank(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)) == 2
    assert gf2_rank(np.zeros((3, 5), dtype=np.uint8)) == 0


def test_plumbing_hypergraph_product():
    assert css_parameters(*hypergraph_product(repetition_code(3), repetition_code(3))) == (13, 1)
    for d in (3, 4, 5):
        HX, HZ = hypergraph_product(cycle_code(d), cycle_code(d))
        assert not (HX.astype(int) @ HZ.T.astype(int) % 2).any()
        assert css_parameters(HX, HZ) == (2 * d * d, 2)


# ---------------------------------------------------------------------------
# TODO 1: toric_checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", DS)
def test_checks_shape_dtype_weights(d):
    H_Z, H_X = toric_checks(d)
    n = 2 * d * d
    for H in (H_Z, H_X):
        assert H.shape == (d * d, n)
        assert H.dtype == np.uint8
        assert set(np.unique(H)) <= {0, 1}
        assert (H.sum(axis=1) == 4).all(), "every vertex and every face touches 4 edges"
        assert (H.sum(axis=0) == 2).all(), "every edge has 2 endpoints and borders 2 faces"


@pytest.mark.parametrize("d", DS)
def test_checks_follow_the_indexing_convention(d):
    H_Z, H_X = toric_checks(d)
    for x in range(d):
        for y in range(d):
            star = {hedge_index(d, x, y), hedge_index(d, x - 1, y),
                    vedge_index(d, x, y), vedge_index(d, x, y - 1)}
            assert set(vector_to_support(H_Z[vertex_index(d, x, y)])) == star
            boundary = {hedge_index(d, x, y), vedge_index(d, x + 1, y),
                        hedge_index(d, x, y + 1), vedge_index(d, x, y)}
            assert set(vector_to_support(H_X[face_index(d, x, y)])) == boundary


@pytest.mark.parametrize("d", DS)
def test_checks_css_condition_and_k(d):
    H_Z, H_X = toric_checks(d)
    assert not (H_X.astype(int) @ H_Z.T.astype(int) % 2).any(), "H_X H_Z^T must vanish mod 2"
    assert gf2_rank(H_Z) == d * d - 1
    assert gf2_rank(H_X) == d * d - 1
    assert css_parameters(H_X, H_Z) == (2 * d * d, 2)


# ---------------------------------------------------------------------------
# TODO 2: toric_logicals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", DS)
def test_logicals_supports(d):
    L_X, L_Z = toric_logicals(d)
    n = 2 * d * d
    assert L_X.shape == (2, n) and L_Z.shape == (2, n)
    assert L_X.dtype == np.uint8 and L_Z.dtype == np.uint8
    assert set(vector_to_support(L_X[0])) == {hedge_index(d, x, 0) for x in range(d)}
    assert set(vector_to_support(L_X[1])) == {vedge_index(d, 0, y) for y in range(d)}
    assert set(vector_to_support(L_Z[0])) == {hedge_index(d, 0, y) for y in range(d)}
    assert set(vector_to_support(L_Z[1])) == {vedge_index(d, x, 0) for x in range(d)}


@pytest.mark.parametrize("d", DS)
def test_logicals_commute_and_pair(d):
    H_Z, H_X = toric_checks(d)
    L_X, L_Z = toric_logicals(d)
    assert not (H_Z.astype(int) @ L_X.T.astype(int) % 2).any(), "L_X rows must be cycles"
    assert not (H_X.astype(int) @ L_Z.T.astype(int) % 2).any(), "L_Z rows must commute with plaquettes"
    assert ((L_X.astype(int) @ L_Z.T.astype(int)) % 2 == np.eye(2, dtype=int)).all()
    assert gf2_rank(np.vstack([H_X, L_X])) == gf2_rank(H_X) + 2, "L_X must not be stabilizers"


# ---------------------------------------------------------------------------
# TODO 3: coset_test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 5, 6])
def test_coset_outcomes(d):
    H_Z, H_X = toric_checks(d)
    L_X, L_Z = toric_logicals(d)
    n = 2 * d * d
    vec = lambda sup: support_to_vector(n, sup)
    e, twin = plaquette_pair(d)
    zero = np.zeros(n, dtype=np.uint8)

    assert coset_test(vec(e), vec(e), H_Z, L_Z) == "success"
    assert coset_test(vec(e), vec(twin), H_Z, L_Z) == "success"
    assert coset_test(zero, H_X[face_index(d, 2, 1)], H_Z, L_Z) == "success"
    assert coset_test(zero, L_X[0], H_Z, L_Z) == "logical"
    assert coset_test(zero, L_X[1], H_Z, L_Z) == "logical"
    assert coset_test(vec(e), vec(twin) ^ L_X[1], H_Z, L_Z) == "logical"
    assert coset_test(zero, H_X[0] ^ L_X[0] ^ L_X[1], H_Z, L_Z) == "logical"
    assert coset_test(vec(e), vec(e[:1]), H_Z, L_Z) == "invalid"
    assert coset_test(zero, vec((0,)), H_Z, L_Z) == "invalid"
    assert coset_test(vec(e), None, H_Z, L_Z) == "invalid"


# ---------------------------------------------------------------------------
# TODO 4: min_weight_table
# ---------------------------------------------------------------------------


def bfs_min_weights(H_Z, w_max):
    """Independent check: minimum weight of every syndrome by breadth-first
    search over syndromes, adding one column at a time."""
    cols = [syndrome_key(H_Z[:, j]) for j in range(H_Z.shape[1])]
    as_int = lambda k: int.from_bytes(k, "big")
    col_ints = [as_int(c) for c in cols]
    start = 0
    dist = {start: 0}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if dist[s] == w_max:
            continue
        for c in col_ints:
            t = s ^ c
            if t not in dist:
                dist[t] = dist[s] + 1
                queue.append(t)
    return dist, as_int


@pytest.mark.parametrize("d,w_max", [(3, 3), (4, 2), (5, 2)])
def test_table_is_minimum_weight(d, w_max):
    H_Z, _ = toric_checks(d)
    n = 2 * d * d
    table = min_weight_table(H_Z, w_max)
    dist, as_int = bfs_min_weights(H_Z, w_max)
    assert len(table) == len(dist), "one entry per syndrome reachable with weight <= w_max"
    for key, sup in table.items():
        assert isinstance(sup, tuple) and list(sup) == sorted(sup)
        assert syndrome_key(syndrome_of(H_Z, support_to_vector(n, sup))) == key
        assert len(sup) == dist[as_int(key)], "stored explanation is not of minimum weight"


@pytest.mark.parametrize("d", [3, 5])
def test_table_tie_break(d):
    H_Z, _ = toric_checks(d)
    n = 2 * d * d
    table = min_weight_table(H_Z, 2)
    assert table[syndrome_key(np.zeros(d * d, dtype=np.uint8))] == ()
    e, twin = plaquette_pair(d)
    assert e == (0, d * d + 1) and twin == (d, d * d)
    s = syndrome_of(H_Z, support_to_vector(n, e))
    assert table[syndrome_key(s)] == e, "first in itertools.combinations order wins"
    for j in (0, 7, n - 1):
        s = syndrome_of(H_Z, support_to_vector(n, (j,)))
        assert table[syndrome_key(s)] == (j,)


# ---------------------------------------------------------------------------
# TODO 5: decoding_radius
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [3, 4, 5])
def test_radius_of_minimum_weight_decoding(d):
    H_Z, _ = toric_checks(d)
    _, L_Z = toric_logicals(d)
    n = 2 * d * d
    t = (d - 1) // 2
    decode = make_table_decoder(min_weight_table(H_Z, t + 1), n)
    rho, witness = decoding_radius(decode, H_Z, L_Z, t + 1)
    assert rho == t, f"minimum weight decoding on T_{d} has radius floor((d-1)/2) = {t}"
    assert witness is not None and len(witness) == t + 1
    e = support_to_vector(n, witness)
    assert coset_test(e, decode(syndrome_of(H_Z, e)), H_Z, L_Z) == "logical"


def test_radius_without_failures_returns_lower_bound():
    d = 5
    H_Z, _ = toric_checks(d)
    _, L_Z = toric_logicals(d)
    decode = make_table_decoder(min_weight_table(H_Z, 2), 2 * d * d)
    assert decoding_radius(decode, H_Z, L_Z, 2) == (2, None)
    assert decoding_radius(decode, H_Z, L_Z, 3, keep=lambda sup: len(sup) < 3) == (3, None)


def test_radius_counts_a_give_up_as_failure():
    d = 3
    H_Z, _ = toric_checks(d)
    _, L_Z = toric_logicals(d)
    assert decoding_radius(lambda s: None, H_Z, L_Z, 2) == (0, (0,))


def test_radius_respects_keep():
    d = 5
    H_Z, _ = toric_checks(d)
    _, L_Z = toric_logicals(d)
    n = 2 * d * d
    e, _ = plaquette_pair(d)
    seen = []

    def decode(s):
        return support_to_vector(n, e)  # right only for the plaquette syndrome

    def keep(sup):
        seen.append(sup)
        return sup == e

    assert decoding_radius(decode, H_Z, L_Z, 2, keep=keep) == (2, None)
    assert len(seen) == n + n * (n - 1) // 2, "keep must be asked about every support"


# ---------------------------------------------------------------------------
# Supplied: is_degenerate (du Crest Definition 2)
# ---------------------------------------------------------------------------


def test_is_degenerate_examples():
    d = 5
    H_Z, _ = toric_checks(d)
    e, twin = plaquette_pair(d)
    assert is_degenerate(e, H_Z) and is_degenerate(twin, H_Z)
    assert not is_degenerate((hedge_index(d, 0, 0),), H_Z)
    assert not is_degenerate((hedge_index(d, 0, 0), hedge_index(d, 2, 2)), H_Z)
