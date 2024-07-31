"""Contains global environment variables."""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
DATASETS_DIR = os.path.join(ROOT_DIR, 'datasets')