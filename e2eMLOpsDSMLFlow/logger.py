# e2eMLOpsDSMLFlow/logger.py

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s",
)

def get_logger(name):
    return logging.getLogger(name)
