import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
RAW_DATA = os.path.join(DATA_DIR, "raw", "actaruslab_f1_telemetry_2026.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reports")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALID_SIZE = 0.15

STABILITY_BINS = [0, 30, 60, 95, 101]
STABILITY_LABELS = ["severe", "moderate", "mild", "stable"]

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
EDA_FIGURES_DIR = os.path.join(FIGURES_DIR, "eda")
MODELS_FIGURES_DIR = os.path.join(FIGURES_DIR, "models")

MODELS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "models")
DEEP_ENSEMBLE_DIR = os.path.join(MODELS_OUTPUT_DIR, "deep_ensemble")
MODELS_COMPARISON_CSV = os.path.join(REPORTS_DIR, "model_comparison.csv")

TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
VALID_CSV = os.path.join(PROCESSED_DIR, "valid.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")

FEATURE_COLS = ["speed_kmh", "wing_angle_deg", "drs_active", "downforce_n", "drag_n"]

NN_BATCH_SIZE = 256
NN_MAX_EPOCHS = 200
NN_EARLY_STOP_PATIENCE = 20
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-5

MLP_HIDDEN_UNITS = [128, 256, 128, 64]
MLP_DROPOUT = 0.2

DEEP_ENSEMBLE_M = 3

TABNET_N_D = 16
TABNET_N_A = 16
TABNET_N_STEPS = 3
TABNET_GAMMA = 1.3
TABNET_LR = 2e-2
TABNET_MAX_EPOCHS = 150

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LR = 0.05

INFERENCE_WARMUP = 100
INFERENCE_REPEATS = 1000

DTYPE = "float32"

# ==== PSO Configuration ====
PSO_N_PARTICLES = 50
PSO_W = 0.7
PSO_W_START = 0.9
PSO_W_END = 0.4
PSO_W_ALPHA = 1.0
PSO_C1 = 2.0
PSO_C2 = 2.0
PSO_MAX_ITER = 100
PSO_EARLY_STOP_ITERS = 12
PSO_TOL = 1e-6

PSO_FIGURES_DIR = os.path.join(FIGURES_DIR, "pso")

# ==== High-Performance Region Discovery ====
HPR_THRESHOLDS = [0.99, 0.95, 0.90]           # fraction-of-gbest thresholds
HPR_ABSOLUTE_THRESHOLD = 99.0                   # absolute fitness threshold (stability_index)
HPR_PERCENTILES = [0.01, 0.05, 0.10]            # top-P% filtering
HPR_CLUSTER_METHOD = "dbscan"                   # "kmeans" or "dbscan" or "hdbscan"
HPR_CLUSTER_N = None                            # n_clusters for kmeans (None=auto via silhouette)
HPR_DBSCAN_EPS = None                           # None=auto via k-distance
HPR_DBSCAN_MIN_SAMPLES = 5
HPR_N_TRIALS = 30                               # independent PSO runs for region discovery
HPR_REGIONS_FIGURES_DIR = os.path.join(FIGURES_DIR, "regions")

# Parameter bounds for F1 aerodynamic optimisation
# [speed_kmh, wing_angle_deg, drs_active, downforce_n, drag_n]
PSO_PARAM_BOUNDS = [
    [80.0, 360.0],
    [0.0, 45.0],
    [0.0, 1.0],
    [0.0, 10000.0],
    [0.0, 600.0],
]
PSO_DISCRETE_INDICES = [2]  # drs_active

# Risk-sensitive PSO
PSO_RISK_LAMBDA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
PSO_DENSITY_K = 10
PSO_DENSITY_LAMBDA = 0.5

AL_TOP_K = 20
AL_NN_NEIGHBORS = 5
AL_PSO_INTERVAL = 10
AL_MAX_ROUNDS = 5
AL_FINETUNE_EPOCHS = 30
AL_FINETUNE_LR = 5e-4

AL_FIGURES_DIR = os.path.join(FIGURES_DIR, "active_learning")
AL_REPORT_PATH = os.path.join(REPORTS_DIR, "active_learning_report.md")

# ==== Multi-Scenario Configuration ====
SCENARIOS_DIR = os.path.join(OUTPUTS_DIR, "scenarios")
SCENARIOS_FIGURES_DIR = os.path.join(FIGURES_DIR, "scenarios")
SCENARIO_N_TRIALS = 15
SCENARIO_N_PARTICLES = 50
SCENARIO_MAX_ITER = 80
SCENARIO_SEED_BASE = 4200
SCENARIO_LAMBDA_RISK = 1.0
SCENARIO_PENALTY_DRAG = 0.05
SCENARIO_PENALTY_DOWNFORCE = 0.01
SCENARIO_SENSITIVITY_DELTA = 0.10
