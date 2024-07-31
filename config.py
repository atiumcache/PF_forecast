"""Contains global environment variables."""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
PF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pf_avg_betas")
TREND_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "trend_forecast")
HOSP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "hosp_forecast")

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
