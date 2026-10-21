#!/bin/bash
# Builds the venv CLIP-PCQA runs in: VAPE's environment through --system-site-packages (torch,
# torchvision and VAPE itself) and the packages of venv_requirements.txt. VAPE's environment
# does not change. pip honours PIP_CONSTRAINT, which pins the torch stack in VAPE's container.
#
#   usage: setup_venv.sh <python of VAPE's environment> <venv directory>
set -euo pipefail
fork=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
base_python=$1
venv=$2
"$base_python" -m venv --system-site-packages "$venv"
"$venv/bin/python" -m pip install --requirement "$fork/venv_requirements.txt"
"$venv/bin/python" -m pip check
