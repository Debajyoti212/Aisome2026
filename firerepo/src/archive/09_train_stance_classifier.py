"""
ARCHIVED EXPERIMENT

This script documents an earlier classifier configuration.
It was superseded by 12_train_run.py and was not used to
produce the final submitted Models 1, 2, or 3.

It is retained for research transparency.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Archived experiment: Preliminary stance classifier training (v1).")


if __name__ == "__main__":
    main()
