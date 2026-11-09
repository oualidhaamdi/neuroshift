#!/usr/bin/env bash
# NeuroShift - minimal FSL install inside WSL2 Ubuntu (or native Linux), via conda.
#
# Installs miniforge3 into $HOME and the FSL packages needed by the pipeline
# (bet, flirt, fast) from the public FSL conda channel. The pipeline's
# check_fsl_availability() looks for them in $HOME/miniforge3/bin/.
#
# NOTE: FSL is distributed under a NON-COMMERCIAL licence
#       (https://fsl.fmrib.ox.ac.uk/fsl/docs/#/license). Make sure your use
#       complies with it. The pipeline works without FSL (SimpleITK fallback).
#
# Usage (from inside WSL/Linux, not from PowerShell):
#   bash scripts/setup_fsl_wsl.sh

set -euo pipefail

MINIFORGE_DIR="$HOME/miniforge3"
FSL_CHANNEL="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public"

echo "=== NeuroShift FSL setup ==="
echo "  Target conda: $MINIFORGE_DIR"

# 1. Nothing to do if FSL tools are already present
if [ -f "$MINIFORGE_DIR/bin/bet" ] && [ -f "$MINIFORGE_DIR/bin/flirt" ] && [ -f "$MINIFORGE_DIR/bin/fast" ]; then
    echo "  [OK] FSL already present in $MINIFORGE_DIR/bin (bet/flirt/fast)."
    exit 0
fi

# 2. System dependencies for the download
echo "  Installing system dependencies (sudo)..."
sudo apt-get update -qq
sudo apt-get install -y -qq wget bzip2 ca-certificates

# 3. Install miniforge3 if missing
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
    echo "  Downloading and installing miniforge3..."
    ARCH="$(uname -m)"   # x86_64 or aarch64
    wget -q "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
    rm -f /tmp/miniforge.sh
    echo "  [OK] miniforge3 installed."
else
    echo "  [OK] miniforge3 already present."
fi

# 4. Install FSL packages from the public channel
echo "  Installing FSL packages (this may take a few minutes)..."
"$MINIFORGE_DIR/bin/conda" install -y \
    -c "$FSL_CHANNEL" \
    -c conda-forge \
    fsl-bet2 fsl-flirt fsl-fast4 fsl-avwutils

# 5. Verify
echo ""
echo "=== Verification ==="
ok=1
for exe in bet flirt fast; do
    if [ -f "$MINIFORGE_DIR/bin/$exe" ]; then
        echo "  [OK] $exe -> $MINIFORGE_DIR/bin/$exe"
    else
        echo "  [FAIL] $exe not found!"
        ok=0
    fi
done

if [ "$ok" -eq 1 ]; then
    echo ""
    echo "FSL ready. The pipeline will detect it automatically."
else
    echo ""
    echo "Something went wrong: check the conda log above."
    exit 1
fi
