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
            Penalties are SUBTRACTED from fitness (i.e. higher penalty → lower fitness).
        """
        self.name = name
        self.code = code
        self.description = description
        self._penalty_terms = penalty_terms or []

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
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIO_S1_MONZA = ScenarioDefinition(
    name="S1 Monza High-Speed",
    code="S1_monza",
    description="High-speed circuit: speed > 280, low wing angle, DRS active, minimise drag",
    bounds_overrides={
        0: [280.0, 360.0],   # speed_kmh: high only
        1: [0.0, 15.0],      # wing_angle_deg: low
    },
    fixed_params={
        2: 1,                 # drs_active: always on
    },
    penalty_terms=[_drag_penalty_fn(weight=SCENARIO_PENALTY_DRAG)],
)

SCENARIO_S2_MONACO = ScenarioDefinition(
    name="S2 Monaco High-Downforce",
    code="S2_monaco",
    description="Street circuit: low speed, high wing angle, no DRS, downforce > 3000N",
    bounds_overrides={
        0: [80.0, 200.0],    # speed_kmh: low only
        1: [20.0, 40.0],     # wing_angle_deg: high
        3: [3000.0, 10000.0], # downforce_n: high
    },
    fixed_params={
        2: 0,                 # drs_active: always off
    },
    penalty_terms=[_downforce_penalty_fn(threshold=3000.0, weight=SCENARIO_PENALTY_DOWNFORCE)],
)

SCENARIO_S3_BALANCED = ScenarioDefinition(
    name="S3 Balanced (Optional)",
    code="S3_balanced",
    description="Medium-speed balanced setup: 200 < v < 280, moderate wing angle, DRS free",
    bounds_overrides={
        0: [200.0, 280.0],   # speed_kmh: medium
        1: [10.0, 30.0],     # wing_angle_deg: medium
    },
)

SCENARIO_S4_WET = ScenarioDefinition(
    name="S4 Wet/Inter (Optional)",
    code="S4_wet",
    description="Wet conditions: moderate speed, high wing angle, no DRS, stability priority",
    bounds_overrides={
        0: [100.0, 250.0],   # speed_kmh: moderate-max
        1: [15.0, 35.0],     # wing_angle_deg: moderately high
    },
    fixed_params={
        2: 0,                 # drs_active: off for wet safety
    },
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
