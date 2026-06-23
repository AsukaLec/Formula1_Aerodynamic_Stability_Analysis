import numpy as np
from copy import deepcopy
from src.utils.config import (
    PSO_PARAM_BOUNDS, PSO_DISCRETE_INDICES,
    SCENARIO_PENALTY_DRAG, SCENARIO_PENALTY_DOWNFORCE,
)

FEATURE_COLS = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]


class ScenarioDefinition:
    """Defines a single F1 track scenario with parameter bounds and soft constraints."""

    def __init__(
        self,
        name,
        code,
        description,
        bounds_overrides=None,
        fixed_params=None,
        penalty_terms=None,
        weights=None,
    ):
        """
        Parameters
        ----------
        name : str
            Human-readable name (e.g. "S1 Monza High-Speed").
        code : str
            Short code for file naming (e.g. "S1_monza").
        description : str
            Engineering description of the scenario.
        bounds_overrides : dict or None
            {col_index: [lower, upper]} to override from default PSO_PARAM_BOUNDS.
        fixed_params : dict or None
            {col_index: value} — forces parameter to a single value.
        penalty_terms : list of callable or None
            Each callable takes (N, dim) positions array, returns (N,) penalty values.
        weights : dict or None
            Multi-objective weights: {"w_stability": float, "w_efficiency": float, "w_power": float}.
            Powers the FitnessMultiObjective composite score. If None, falls back to single-objective.
        """
        self.name = name
        self.code = code
        self.description = description
        self._penalty_terms = penalty_terms or []
        self.weights = weights or {}

        default_bounds = np.array(PSO_PARAM_BOUNDS, dtype=np.float64)
        self._bounds = default_bounds.copy()
        if bounds_overrides:
            for idx, (lb, ub) in bounds_overrides.items():
                self._bounds[idx] = [lb, ub]

        self._fixed_params = fixed_params or {}
        for idx, val in self._fixed_params.items():
            self._bounds[idx] = [val, val]

        self.discrete_indices = list(PSO_DISCRETE_INDICES)

    @property
    def bounds(self):
        return self._bounds.copy()

    @property
    def fixed_params(self):
        return deepcopy(self._fixed_params)

    def penalize(self, positions):
        """Compute total penalty for given positions. Returns (N,) array."""
        penalties = np.zeros(len(positions), dtype=np.float64)
        for fn in self._penalty_terms:
            penalties += np.asarray(fn(positions), dtype=np.float64)
        return penalties

    def __repr__(self):
        info = [f"Scenario: {self.name} ({self.code})"]
        info.append(f"  {self.description}")
        info.append(f"  bounds:  {dict(zip(FEATURE_COLS, self._bounds.tolist()))}")
        if self._fixed_params:
            info.append(f"  fixed:   {dict(zip([FEATURE_COLS[i] for i in self._fixed_params], self._fixed_params.values()))}")
        return "\n".join(info)


# ---------------------------------------------------------------------------
# Penalty term factories
# ---------------------------------------------------------------------------

def _drag_penalty_fn(weight=SCENARIO_PENALTY_DRAG, drag_max=600.0):
    """Penalise high drag: penalty = weight * (drag / drag_max)."""
    def fn(positions):
        drag = positions[:, 4]  # drag_n at column index 4
        return weight * (drag / drag_max)
    return fn


def _downforce_penalty_fn(threshold=3000.0, weight=SCENARIO_PENALTY_DOWNFORCE):
    """Penalise downforce below threshold: penalty = weight * max(0, threshold - df)."""
    def fn(positions):
        df = positions[:, 3]  # downforce_n at column index 3
        deficit = np.maximum(0, threshold - df)
        return weight * deficit
    return fn


# ---------------------------------------------------------------------------
# Scenario definitions (v2: multi-objective with fixed max speed)
# Training data ranges: speed[100,365], wing[5,35], downforce[105,8979], drag[18,516]
# All scenario bounds are clipped to training P1-P99 for OOD safety.
# ---------------------------------------------------------------------------

TRAIN_WING_MIN = 20.0     # engineering minimum — wing never below 20° in real F1
TRAIN_WING_MAX = 35.0     # training data max
TRAIN_DF_MIN = 105.0      # training data min
TRAIN_DF_MAX = 8979.0     # training data max (~P99)
TRAIN_DRAG_MIN = 18.0     # training data min
TRAIN_DRAG_MAX = 516.0    # training data max

SCENARIO_S1_MONZA = ScenarioDefinition(
    name="S1 Monza High-Speed",
    code="S1_monza",
    description="Top speed ~345 km/h, minimal wing angle, DRS open, efficiency-critical",
    bounds_overrides={
        0: [345.0, 345.0],         # speed_kmh: FIXED at top speed
        1: [TRAIN_WING_MIN, 29.0], # wing_angle: physical constraint at 345 km/h
        3: [TRAIN_DF_MIN, TRAIN_DF_MAX],
        4: [TRAIN_DRAG_MIN, TRAIN_DRAG_MAX],
    },
    fixed_params={
        2: 1,  # drs_active: always on
    },
    weights={"w_stability": 0.4, "w_efficiency": 0.3, "w_power": 0.3},
)

SCENARIO_S2_MONACO = ScenarioDefinition(
    name="S2 Monaco High-Downforce",
    code="S2_monaco",
    description="Top speed ~300 km/h, high wing angle, no DRS, downforce priority",
    bounds_overrides={
        0: [300.0, 300.0],              # speed_kmh: FIXED
        1: [TRAIN_WING_MIN, TRAIN_WING_MAX],  # wing: full range feasible
        3: [TRAIN_DF_MIN, TRAIN_DF_MAX],
        4: [TRAIN_DRAG_MIN, TRAIN_DRAG_MAX],
    },
    fixed_params={
        2: 0,  # drs_active: always off
    },
    weights={"w_stability": 0.5, "w_efficiency": 0.4, "w_power": 0.1},
)

SCENARIO_S3_BALANCED = ScenarioDefinition(
    name="S3 Balanced",
    code="S3_balanced",
    description="Top speed ~320 km/h, moderate wing, DRS free, balanced efficiency/stability",
    bounds_overrides={
        0: [320.0, 320.0],
        1: [TRAIN_WING_MIN, TRAIN_WING_MAX],
        3: [TRAIN_DF_MIN, TRAIN_DF_MAX],
        4: [TRAIN_DRAG_MIN, TRAIN_DRAG_MAX],
    },
    weights={"w_stability": 0.4, "w_efficiency": 0.4, "w_power": 0.2},
)

SCENARIO_S4_WET = ScenarioDefinition(
    name="S4 Wet",
    code="S4_wet",
    description="Top speed ~290 km/h, high wing angle, no DRS, stability above all",
    bounds_overrides={
        0: [290.0, 290.0],
        1: [TRAIN_WING_MIN, TRAIN_WING_MAX],
        3: [TRAIN_DF_MIN, TRAIN_DF_MAX],
        4: [TRAIN_DRAG_MIN, TRAIN_DRAG_MAX],
    },
    fixed_params={
        2: 0,  # drs_active: off for wet safety
    },
    weights={"w_stability": 0.6, "w_efficiency": 0.2, "w_power": 0.2},
)

# Master list
ALL_SCENARIOS = [SCENARIO_S1_MONZA, SCENARIO_S2_MONACO]
OPTIONAL_SCENARIOS = [SCENARIO_S3_BALANCED, SCENARIO_S4_WET]


def get_scenario(code):
    """Retrieve scenario by code string."""
    for s in ALL_SCENARIOS + OPTIONAL_SCENARIOS:
        if s.code == code:
            return s
    raise KeyError(f"Unknown scenario code: {code}")
