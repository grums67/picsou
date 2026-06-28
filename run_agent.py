#!/usr/bin/env python3
"""Picsou agent runner - starts the continuous trading loop."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_config
from src.picsou import PicsouAgent, setup_logging


def main():
    config = get_config()
    setup_logging(config.paths.logs)

    agent = PicsouAgent(config)
    agent.run()  # runs forever with config.loop_interval (default 300s = 5min)


if __name__ == "__main__":
    main()