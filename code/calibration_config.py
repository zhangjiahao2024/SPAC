"""
    The same calibration configuration for CF + VT-CPFM-1 experiments.

    All calibration methods (SAPC, NSGA-II, MOEA-D, profiled variants, ablations, etc.) should import
    parameter names/bounds and simulator constraints from this module so the only 
    methodological difference is the search strategy.

"""
from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Car-following model parameter spaces
# -----------------------------------------------------------------------------
CFM_CONFIG = {
    "idm": {
        "code": 0,
        "param_names": ("s0", "T", "a_max", "b", "v_max", "delta"),
        "lb": np.array([1.0, 0.1, 0.1, 0.1, 5.0, 1.0], dtype=np.float64),
        "ub": np.array([5.0, 3.0, 5.0, 5.0, 33.0, 10.0], dtype=np.float64),
    },
    "ghr": {   # a = c * max(v, GHR_V_FLOOR)^m * dv / max(s, 0.01)^l  (no reaction time); bounds: Zhu et al. (2020)
        "code": 1,
        "param_names": ("c", "m", "l"),
        "lb": np.array([0.0, -10.0, 0.0], dtype=np.float64),
        "ub": np.array([60.0, 10.0, 10.0], dtype=np.float64),
    },
    "gipps": {
        "code": 2,
        "param_names": ("tau", "theta", "a_max", "b", "s0", "v_max"),
        "lb": np.array([0.1, 0.1, 0.1, 0.1, 1.0, 5.0], dtype=np.float64),
        "ub": np.array([2.0, 1.0, 5.0, 5.0, 5.0, 33.0], dtype=np.float64),
    },
    "ovm": {
        "code": 3,
        "param_names": ("v1", "v2", "c1", "c2", "kv"),
        "lb": np.array([0.0, 10.0, 0.05, 1.0, 0.1], dtype=np.float64),
        "ub": np.array([5.0, 30.0, 1.0, 10.0, 2.0], dtype=np.float64),
    },
}

FUEL_CONFIG = {
    "param_names": ("alpha0", "alpha1", "alpha2"),
    "lb": np.array([1e-5, 1e-6, 1e-7], dtype=np.float64),
    "ub": np.array([1e-2, 1e-3, 1e-5], dtype=np.float64),
}

# -----------------------------------------------------------------------------
# Physical simulation constants
# -----------------------------------------------------------------------------
ACC_MIN = -6.0
ACC_MAX = 4.0
COLLISION_SPACING_PENALTY = 100.0  # simulated spacing after collision (m)

# VT-CPFM-1 constants
RHO = 1.2256
C_H = 0.982
C_R = 1.75
ROLL_C1 = 0.0328
ROLL_C2 = 4.575
GRAVITY = 9.8066
MASS_FACTOR = 1.04

# -----------------------------------------------------------------------------
# Shared spectral/SAPC settings
# -----------------------------------------------------------------------------
FS = 30.0
F_MIN = 0.02
F_MAX = 2.0
ENERGY_THRESHOLD = 0.85
LF_CUTOFF_DEFAULT = 0.30

LF_SEQ_LEN = 90
PSD_BINS = 64
HF_TOKEN_DIM = 5
CONTEXT_DIM = 13
EMBED_DIM = 64
N_HEADS = 4
FCPG_LR = 3e-4
SIGMA_MIN = 0.03
SIGMA_MAX = 0.50

GUIDED_RATIO = 0.80
REPLAY_BATCH_SIZE = 96
FCPG_STEPS_PER_ITER = 1
PREF_LAMBDAS = np.linspace(0.05, 0.95, 10, dtype=np.float32)
TCHEBY_RHO = 0.05

DEFAULT_EVAL_BUDGET = 20000
DEFAULT_N_CANDIDATES = 100
DEFAULT_CHECKPOINTS = (1000, 2000, 5000, 10000, 15000, 20000)

GHR_V_FLOOR = 1.0


def get_model_config(cfm: str):
    """
        Return a copy-safe config dictionary for one CF model.
    """
    key = str(cfm).lower()
    if key not in CFM_CONFIG:
        raise KeyError(f"Unknown CFM '{cfm}'. Choose from {tuple(CFM_CONFIG)}")
    base = CFM_CONFIG[key]
    cf_lb = base["lb"].copy()
    cf_ub = base["ub"].copy()
    fuel_lb = FUEL_CONFIG["lb"].copy()
    fuel_ub = FUEL_CONFIG["ub"].copy()
    return {
        "name": key,
        "code": int(base["code"]),
        "cf_param_names": tuple(base["param_names"]),
        "fuel_param_names": tuple(FUEL_CONFIG["param_names"]),
        "param_names": tuple(base["param_names"]) + tuple(FUEL_CONFIG["param_names"]),
        "cf_lb": cf_lb,
        "cf_ub": cf_ub,
        "fuel_lb": fuel_lb,
        "fuel_ub": fuel_ub,
        "lb": np.concatenate([cf_lb, fuel_lb]),
        "ub": np.concatenate([cf_ub, fuel_ub]),
        "cf_dim": len(cf_lb),
        "param_dim": len(cf_lb) + len(fuel_lb),
    }
