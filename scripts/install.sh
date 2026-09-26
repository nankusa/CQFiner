#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python -r requirements.txt -c requirements-lock.txt
