#!/usr/bin/env bash
set -euo pipefail

if ! command -v octave >/dev/null 2>&1; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends octave
  rm -rf /var/lib/apt/lists/*
fi

exec python3 tools/eval/run_tnl2k_experiment.py "$@"
