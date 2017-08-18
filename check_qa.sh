#!/bin/bash

# QA checks: run this before every commit

set -e
flake8 --exclude='ipython_log.py*' .
isort --recursive --check-only --skip='ipython_log.py' .
