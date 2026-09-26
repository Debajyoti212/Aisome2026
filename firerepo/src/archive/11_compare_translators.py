"""
ARCHIVED EXPERIMENT

This script contains the translator-comparison experiment
reported in the paper.

The experiment was preliminary and should not be interpreted
as a controlled comparison of classifiers because the
associated classifier configuration was not held constant.

It is retained for research transparency.
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Archived experiment: Translator comparison (Custom NMT vs NLLB).")


if __name__ == "__main__":
    main()
