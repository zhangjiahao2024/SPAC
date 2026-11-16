"""
    Baseline search strategies sharing the same simulator, objectives and budget.
"""
from __future__ import annotations

import numpy as np

from sapc_v2 import EvalRecorder


# ---------------------------------------------------------------------------
# variation operators on [0,1]^d
# ---------------------------------------------------------------------------
def sbx(P1, P2, rng, eta=15.0, pc=0.9):
    n, d = P1.shape
    C1, C2 = P1.copy(), P2.copy()
    do = (rng.random(n) < pc)[:, None] & (rng.random((n, d)) < 0.5)
    u = rng.random((n, d))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u) + 1e-12)) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * P1 + (1 - beta) * P2)
    c2 = 0.5 * ((1 - beta) * P1 + (1 + beta) * P2)
    C1[do], C2[do] = c1[do], c2[do]
    return np.clip(C1, 0, 1), np.clip(C2, 0, 1)


def poly_mutation(X, rng, eta=20.0, pm=None):
    n, d = X.shape
    pm = 1.0 / d if pm is None else pm
    m = rng.random((n, d)) < pm
    u = rng.random((n, d))
    d1, d2 = X, 1 - X
    lo = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    hi = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
    delta = np.where(u < 0.5, lo, hi)
    Y = X.copy()
    Y[m] = X[m] + delta[m]
    return np.clip(Y, 0, 1)


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------
def nd_rank(F):
    n = len(F)
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = le & lt                       # dom[i, j]: i dominates j
    count = dom.sum(0)
    rank = np.full(n, -1)
    r = 0
    front = np.where(count == 0)[0]
    while len(front):
        rank[front] = r
        count = count - dom[front].sum(0)
        count[rank >= 0] = -1
        front = np.where(count == 0)[0]
        r += 1
    return rank


def crowding(F, rank):
    cd = np.zeros(len(F))
    for r in np.unique(rank):
        idx = np.where(rank == r)[0]
        if len(idx) <= 2:
            cd[idx] = np.inf
            continue
        for j in range(F.shape[1]):
            o = idx[np.argsort(F[idx, j])]
            span = F[o[-1], j] - F[o[0], j] + 1e-12
            cd[o[0]] = cd[o[-1]] = np.inf
            cd[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return cd


def _tournament(rank, cd, rng, n):
    a, b = rng.integers(0, len(rank), n), rng.integers(0, len(rank), n)
    better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (cd[a] > cd[b]))
    return np.where(better, a, b)


def constrained_rank(F, viol):
    """
        Deb's constrained domination: feasible first (non-dominated ranks),
        then infeasible ordered by total violation.
    """
    if viol is None or np.all(viol <= 0):
        return nd_rank(F)
    rank = np.empty(len(F), dtype=int)
    feas = viol <= 0
    base = 0
    if feas.any():
        r = nd_rank(F[feas])
        rank[feas] = r
        base = r.max() + 1
    inf = np.where(~feas)[0]
    rank[inf[np.argsort(viol[inf])]] = base + np.arange(len(inf))
    return rank


def nsga2_search(rec: EvalRecorder, rng, pop=100, eta_c=15.0, eta_m=20.0, pc=0.9,
                 violation=None):
    """
        violation(rec, global_idx) -> constraint violation (0 = feasible), optional.
    """
    d = rec.problem.dim
    X = rng.random((pop, d))
    F = rec(X)
    X = X[: len(F)]
    Gi = np.arange(rec.n - len(F), rec.n)
    while not rec.done:
        rank = constrained_rank(F, violation(rec, Gi) if violation else None)
        cd = crowding(F, rank)
        i1 = _tournament(rank, cd, rng, pop // 2)
        i2 = _tournament(rank, cd, rng, pop // 2)
        c1, c2 = sbx(X[i1], X[i2], rng, eta_c, pc)
        Q = poly_mutation(np.vstack([c1, c2]), rng, eta_m)
        FQ = rec(Q)
        Q = Q[: len(FQ)]
        GQ = np.arange(rec.n - len(FQ), rec.n)
        RX, RF, RG = np.vstack([X, Q]), np.vstack([F, FQ]), np.concatenate([Gi, GQ])
        rank = constrained_rank(RF, violation(rec, RG) if violation else None)
        cd = crowding(RF, rank)
        order = np.lexsort((-cd, rank))[:pop]
        X, F, Gi = RX[order], RF[order], RG[order]
    return X, F


# ---------------------------------------------------------------------------
# Single-objective GA (Split, CF stage: minimise spacing NRMSE)
# ---------------------------------------------------------------------------
def ga_minimize(rec: EvalRecorder, rng, pop=100, eta_c=15.0, eta_m=20.0, pc=0.9, objective=0):
    d = rec.problem.dim
    X = rng.random((pop, d))
    f = rec(X)[:, objective]
    
    while not rec.done:
        a, b = rng.integers(0, pop, pop), rng.integers(0, pop, pop)
        par = np.where(f[a] < f[b], a, b)
        c1, c2 = sbx(X[par[: pop // 2]], X[par[pop // 2:]], rng, eta_c, pc)
        Q = poly_mutation(np.vstack([c1, c2]), rng, eta_m)
        fq = rec(Q)[:, objective]
        Q = Q[: len(fq)]
        RX, rf = np.vstack([X, Q]), np.concatenate([f, fq])
        order = np.argsort(rf)[:pop]
        X, f = RX[order], rf[order]
    return X[0], f[0]
