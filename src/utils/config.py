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

DTYPE = "float32"
