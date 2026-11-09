"""NB1 / Lesson 1: the toric code T_d as a decoding problem.

Conventions, fixed for the whole project (Lesson 1 notes, "The lattice T_d"):

    vertex          v(x, y)                          index x + d*y
    horizontal edge h(x, y)  from v(x, y) to v(x+1, y)  index x + d*y
    vertical edge   u(x, y)  from v(x, y) to v(x, y+1)  index d*d + x + d*y
    face            f(x, y)  corners v(x,y), v(x+1,y), v(x+1,y+1), v(x,y+1)
                                                     index x + d*y
    boundary of f(x, y) = h(x, y) + u(x+1, y) + h(x, y+1) + u(x, y)

All coordinates are taken mod d. Qubits sit on the n = 2*d*d edges. We decode
X errors e in F_2^n: the vertex checks H_Z give the syndrome s = H_Z e (mod 2),
and the plaquettes H_X span the X stabilizers. A correction e_hat succeeds iff
e + e_hat lies in the row space of H_X.

Arrays are numpy uint8 with entries 0/1. An error is either a 0/1 vector of
length n or a "support": a sorted tuple of edge indices.

The five functions marked TODO are written by Sippakorn (IEEE AI policy: core
research functions are the author's own). Everything else is supplied plumbing.
Run `pytest tests/test_toric.py -q` until it is green.
"""

from __future__ import annotations

import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Support = Tuple[int, ...]
Decoder = Callable[[np.ndarray], Optional[np.ndarray]]

# ---------------------------------------------------------------------------
# Indexing (supplied)
# ---------------------------------------------------------------------------


def num_qubits(d: int) -> int:
    """n = 2 d^2 edges of T_d."""
    return 2 * d * d


def vertex_index(d: int, x: int, y: int) -> int:
    """Index of vertex v(x, y)."""
    return (x % d) + d * (y % d)


def hedge_index(d: int, x: int, y: int) -> int:
    """Index of the horizontal edge h(x, y) = {v(x, y), v(x+1, y)}."""
    return (x % d) + d * (y % d)


def vedge_index(d: int, x: int, y: int) -> int:
    """Index of the vertical edge u(x, y) = {v(x, y), v(x, y+1)}."""
    return d * d + (x % d) + d * (y % d)


def face_index(d: int, x: int, y: int) -> int:
    """Index of the face f(x, y) whose lower-left corner is v(x, y)."""
    return (x % d) + d * (y % d)


def edge_coords(d: int, e: int) -> Tuple[str, int, int]:
    """Inverse of hedge_index / vedge_index: ('h', x, y) or ('v', x, y)."""
    if not 0 <= e < 2 * d * d:
        raise IndexError(f"edge {e} out of range for T_{d}")
    kind = "h" if e < d * d else "v"
    r = e % (d * d)
    return kind, r % d, r // d


# ---------------------------------------------------------------------------
# Vectors, keys and GF(2) linear algebra (supplied)
# ---------------------------------------------------------------------------


def support_to_vector(n: int, support: Iterable[int]) -> np.ndarray:
    """0/1 uint8 vector of length n with ones on `support` (repeats cancel mod 2)."""
    v = np.zeros(n, dtype=np.uint8)
    for i in support:
        v[i] ^= 1
    return v


def vector_to_support(v: np.ndarray) -> Support:
    """Sorted tuple of the positions where v is 1."""
    return tuple(int(i) for i in np.flatnonzero(np.asarray(v) % 2))


def syndrome_of(H: np.ndarray, e: np.ndarray) -> np.ndarray:
    """H e mod 2 as a uint8 vector."""
    return (H.astype(np.int64) @ np.asarray(e, dtype=np.int64) % 2).astype(np.uint8)


def syndrome_key(s: np.ndarray) -> bytes:
    """Hashable key for a syndrome vector; use it for dictionary lookups."""
    return np.packbits(np.asarray(s, dtype=np.uint8) % 2).tobytes()


def gf2_rank(M: np.ndarray) -> int:
    """Rank of a 0/1 matrix over GF(2) (Gaussian elimination on a copy)."""
    A = (np.array(M, dtype=np.uint8) % 2).copy()
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        pivots = np.flatnonzero(A[r:, c])
        if pivots.size == 0:
            continue
        p = r + int(pivots[0])
        if p != r:
            A[[r, p]] = A[[p, r]]
        hit = np.flatnonzero(A[:, c])
        hit = hit[hit != r]
        A[hit] ^= A[r]
        r += 1
    return r


# ---------------------------------------------------------------------------
# TODO 1: check matrices
# ---------------------------------------------------------------------------


def toric_checks(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (H_Z, H_X) for T_d, both uint8 of shape (d*d, 2*d*d).

    H_Z[vertex_index(d, x, y), e] = 1  iff  edge e has v(x, y) as an endpoint.
    H_X[face_index(d, x, y), e]   = 1  iff  edge e lies on the boundary of f(x, y).

    Use the index helpers above; every coordinate wraps mod d.
    """
    raise NotImplementedError("TODO 1 (NB1): build H_Z and H_X")


# ---------------------------------------------------------------------------
# TODO 2: logical operators
# ---------------------------------------------------------------------------


def toric_logicals(d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (L_X, L_Z), both uint8 of shape (2, 2*d*d).

    L_X row 0: the horizontal loop  h(x, 0) for x = 0..d-1
    L_X row 1: the vertical loop    u(0, y) for y = 0..d-1
    L_Z row 0: the cut              h(0, y) for y = 0..d-1
    L_Z row 1: the cut              u(x, 0) for x = 0..d-1

    They satisfy H_Z L_X^T = 0, H_X L_Z^T = 0 and L_X L_Z^T = I_2 (mod 2).
    """
    raise NotImplementedError("TODO 2 (NB1): build L_X and L_Z")


# ---------------------------------------------------------------------------
# TODO 3: the coset test
# ---------------------------------------------------------------------------


def coset_test(
    e: np.ndarray,
    e_hat: Optional[np.ndarray],
    H_Z: np.ndarray,
    L_Z: np.ndarray,
) -> str:
    """Classify a correction e_hat for the true error e.

    Returns one of
        "success"  e + e_hat is a stabilizer (in the row space of H_X),
        "logical"  e + e_hat is a cycle but not a stabilizer,
        "invalid"  e_hat does not reproduce the syndrome, or e_hat is None
                   (a decoder that gives up returns None).

    Use only H_Z and L_Z; see "The coset test" in the notes for why that is enough.
    """
    raise NotImplementedError("TODO 3 (NB1): the coset test")


# ---------------------------------------------------------------------------
# TODO 4: brute-force minimum-weight decoding table
# ---------------------------------------------------------------------------


def min_weight_table(H_Z: np.ndarray, w_max: int) -> Dict[bytes, Support]:
    """Map every syndrome reachable by an error of weight <= w_max to a
    minimum-weight explanation of it.

    Enumerate supports by weight w = 0, 1, ..., w_max, and within one weight in
    the order of itertools.combinations(range(n), w). Keep the FIRST support
    found for each syndrome; key the dict with syndrome_key(H_Z e mod 2). The
    empty syndrome maps to the empty tuple ().

    This fixes the tie-break: for the plaquette syndrome {v(0,0), v(1,1)} the
    table must return (0, d*d + 1), not (d, d*d).

    Size check: on T_5 with w_max = 3 there are 20,876 supports to try.
    """
    raise NotImplementedError("TODO 4 (NB1): minimum-weight table")


# ---------------------------------------------------------------------------
# TODO 5: exhaustive decoding radius
# ---------------------------------------------------------------------------


def decoding_radius(
    decode: Decoder,
    H_Z: np.ndarray,
    L_Z: np.ndarray,
    w_max: int,
    keep: Optional[Callable[[Support], bool]] = None,
) -> Tuple[int, Optional[Support]]:
    """Exhaustive decoding radius of `decode` up to weight w_max.

    For w = 1, ..., w_max and each support of weight w in
    itertools.combinations order: skip it if `keep` is given and
    keep(support) is False; otherwise run e_hat = decode(H_Z e mod 2) and
    classify it with coset_test.

    Return (w - 1, support) at the first error whose outcome is not "success".
    If nothing fails, return (w_max, None), meaning the radius is at least w_max.

    With keep = "is non-degenerate" the same function gives rho_nd (used in NB5).
    """
    raise NotImplementedError("TODO 5 (NB1): decoding radius")


# ---------------------------------------------------------------------------
# Supplied helpers that build on the TODOs
# ---------------------------------------------------------------------------


def decode_with_table(table: Dict[bytes, Support], s: np.ndarray, n: int) -> Optional[np.ndarray]:
    """Look s up in a min_weight_table; None if it is not in the table."""
    sup = table.get(syndrome_key(s))
    return None if sup is None else support_to_vector(n, sup)


def make_table_decoder(table: Dict[bytes, Support], n: int) -> Decoder:
    """Wrap a table as a decoder s -> e_hat (or None)."""
    return lambda s: decode_with_table(table, s, n)


def is_degenerate(support: Sequence[int], H_Z: np.ndarray) -> bool:
    """du Crest-Mhalla-Savin, Definition 2: e is degenerate iff some e' != e
    with |e'| <= |e| has the same syndrome. Brute force; fine for |e| <= 3 on
    the small lattices used here. No coset condition, exactly as in the paper."""
    n = H_Z.shape[1]
    cols = [int.from_bytes(np.packbits(H_Z[:, j]).tobytes(), "big") for j in range(n)]
    target = 0
    for j in support:
        target ^= cols[j]
    sup = tuple(sorted(support))
    for w in range(len(sup) + 1):
        for cand in itertools.combinations(range(n), w):
            if cand == sup:
                continue
            acc = 0
            for j in cand:
                acc ^= cols[j]
            if acc == target:
                return True
    return False


def failing_counts(decode: Decoder, H_Z: np.ndarray, L_Z: np.ndarray, w_max: int) -> List[int]:
    """F_w for w = 0..w_max: the number of weight-w errors on which decode does
    not succeed (logical or invalid). Uses coset_test."""
    n = H_Z.shape[1]
    counts = []
    for w in range(w_max + 1):
        bad = 0
        for sup in itertools.combinations(range(n), w):
            e = support_to_vector(n, sup)
            if coset_test(e, decode(syndrome_of(H_Z, e)), H_Z, L_Z) != "success":
                bad += 1
        counts.append(bad)
    return counts


def logical_rate_terms(F: Sequence[int], n: int, p: float) -> np.ndarray:
    """The terms F_w p^w (1-p)^(n-w) of the low-noise expansion of P_L."""
    return np.array([F[w] * p**w * (1 - p) ** (n - w) for w in range(len(F))])


def cycle_code(d: int) -> np.ndarray:
    """d x d check matrix of the length-d cycle code: row i checks bits i, i+1 mod d."""
    H = np.zeros((d, d), dtype=np.uint8)
    for i in range(d):
        H[i, i] = 1
        H[i, (i + 1) % d] = 1
    return H


def repetition_code(n: int) -> np.ndarray:
    """(n-1) x n check matrix of the length-n repetition code (open chain)."""
    H = np.zeros((n - 1, n), dtype=np.uint8)
    for i in range(n - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def hypergraph_product(H1: np.ndarray, H2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Tillich-Zemor product, in the notes' convention. Returns (H_X, H_Z):
        H_X = [H1 (x) I_n2 | I_m1 (x) H2^T],  H_Z = [I_n1 (x) H2 | H1^T (x) I_m2].
    """
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    HX = np.hstack([np.kron(H1, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(m1, dtype=np.uint8), H2.T)])
    HZ = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), H2), np.kron(H1.T, np.eye(m2, dtype=np.uint8))])
    return (HX % 2).astype(np.uint8), (HZ % 2).astype(np.uint8)


def css_parameters(H_X: np.ndarray, H_Z: np.ndarray) -> Tuple[int, int]:
    """(n, k) of a CSS code: k = n - rank H_X - rank H_Z."""
    n = H_X.shape[1]
    return n, n - gf2_rank(H_X) - gf2_rank(H_Z)


# ---------------------------------------------------------------------------
# Plotting (supplied)
# ---------------------------------------------------------------------------


def plot_lattice(
    d: int,
    error: Optional[Iterable[int]] = None,
    correction: Optional[Iterable[int]] = None,
    syndrome: Optional[Iterable[int]] = None,
    highlight: Optional[Iterable[int]] = None,
    labels: bool = False,
    title: Optional[str] = None,
    ax=None,
):
    """Draw T_d. Edges and vertices are given as supports (index lists).

    error       red solid edges
    correction  blue dashed edges (drawn on top, offset slightly)
    highlight   green edges (logicals, cuts, stabilizers ...)
    syndrome    black squares on vertices
    labels      write edge indices (useful for d <= 5)

    The right column x = d and top row y = d are pale copies of x = 0 and y = 0:
    the torus is glued there.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(0.9 * d + 1.6, 0.9 * d + 1.6))
    error = set(error or ())
    correction = set(correction or ())
    highlight = set(highlight or ())
    syndrome = set(syndrome or ())

    def seg(e):
        kind, x, y = edge_coords(d, e)
        return ((x, y), (x + 1, y)) if kind == "h" else ((x, y), (x, y + 1))

    for e in range(2 * d * d):
        (x0, y0), (x1, y1) = seg(e)
        ax.plot([x0, x1], [y0, y1], color="0.82", lw=1.2, zorder=1)
        if labels:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, str(e), fontsize=7, color="0.45",
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    for e in highlight:
        (x0, y0), (x1, y1) = seg(e)
        ax.plot([x0, x1], [y0, y1], color="tab:green", lw=4, alpha=0.55, zorder=2)
    for e in error:
        (x0, y0), (x1, y1) = seg(e)
        ax.plot([x0, x1], [y0, y1], color="tab:red", lw=3.2, zorder=3)
    for e in correction:
        (x0, y0), (x1, y1) = seg(e)
        dx, dy = (0, 0.07) if y0 == y1 else (0.07, 0)
        ax.plot([x0 + dx, x1 + dx], [y0 + dy, y1 + dy], color="tab:blue", lw=2.4, ls="--", zorder=4)
    for x in range(d + 1):
        for y in range(d + 1):
            ghost = x == d or y == d
            ax.plot(x, y, "o", ms=4, color="0.75" if ghost else "0.35", zorder=4)
    for v in syndrome:
        x, y = v % d, v // d
        for gx in ([x, d] if x == 0 else [x]):
            for gy in ([y, d] if y == 0 else [y]):
                ghost = gx == d or gy == d
                ax.plot(gx, gy, "s", ms=9, color="0.6" if ghost else "black", zorder=6)
    ax.set_xlim(-0.4, d + 0.4)
    ax.set_ylim(-0.4, d + 0.4)
    ax.set_aspect("equal")
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.tick_params(length=0, labelsize=8)
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
