"""Contains global environment variables."""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
PF_DIR = os.path.join(SRC_DIR, "particle_filter")
BETA_FORECAST_DIR = os.path.join(SRC_DIR, "beta_forecast")
HOSP_FORECAST_DIR = os.path.join(SRC_DIR, "hosp_forecast")

OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
PF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pf_avg_betas")
TREND_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "trend_forecast")
HOSP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "hosp_forecast")

DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

# This SPHERE is only intended for testing on Andrew's local machine.
SPHERE_DIR = "/home/andrew/PycharmProjects/SPHERE/sphere"
