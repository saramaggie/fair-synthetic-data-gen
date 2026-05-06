#!/usr/bin/env bash

RUN_DIR="_run"

source "${RUN_DIR}/setup_venv.sh"
pip install -q -r requirements.txt

jupyter nbconvert --to python Notebook_Models.ipynb --output=Notebook_Models.py --output-dir="${RUN_DIR}"

python "${RUN_DIR}/Notebook_Models.py"