from pathlib import Path

DATA_DIR = Path("data")
PROXY_SCORE_COL = "proxy_score"
LABEL_COL = "labels"

# targets
TARGET_RECALL = 0.9
TARGET_PRECISION = 0.8
CONFIDENCE_LEVEL = 0.95
ORACLE_BUDGET = 100