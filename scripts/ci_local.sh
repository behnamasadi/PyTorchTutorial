#!/usr/bin/env bash
#
# ci_local.sh — run the push-time CI pipeline locally.
#
# Mirrors .github/workflows/ci.yml so you can see green/red BEFORE pushing:
#   1. ensure dependencies (pytest, numpy, torch — CPU-only torch, like the
#      CPU-only GitHub runner; scripts must handle device detection)
#   2. pytest --ignore=projects
#   3. python utils/scripts/verify_scripts.py
#
# Usage:
#   ./scripts/ci_local.sh              # full local CI (also runs on `git push` via pre-push hook)
#   SKIP_INSTALL=1 ./scripts/ci_local.sh   # skip the pip install step
#   ./scripts/ci_local.sh --build      # additionally try `docker build` (mirrors
#                                      # ghcr.yml build step, without pushing; heavy first run)
#
# Exit code is 0 only if every step passes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

pass() { echo "✅ $1"; }
fail() { echo "❌ $1" >&2; }

echo "== [0/3] environment =="
python3 --version
if ! python3 -c "import pytest, numpy, torch" 2>/dev/null; then
  if [ "${SKIP_INSTALL:-0}" = "1" ]; then
    fail "missing dependencies (pytest/numpy/torch) and SKIP_INSTALL=1"
    exit 1
  fi
  echo "Installing missing dependencies (CPU-only torch, as on the CI runner)..."
  python3 -m pip install --upgrade pip
  python3 -m pip install pytest numpy
  # CPU wheel keeps this light; plain `pip install torch` pulls multi-GB CUDA binaries.
  python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
python3 -c "import pytest, numpy; print('pytest', pytest.__version__, '| numpy', numpy.__version__)"
try_torch="$(python3 -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())" 2>&1 || true)"
echo "$try_torch"
pass "environment ready"

echo
echo "== [1/3] pytest --ignore=projects =="
if python3 -m pytest --ignore=projects -q; then
  pass "pytest"
else
  fail "pytest — fix failures before pushing"
  exit 1
fi

echo
echo "== [2/3] verify_scripts.py =="
if python3 utils/scripts/verify_scripts.py > /tmp/ci_local_verify.log 2>&1; then
  tail -n 5 /tmp/ci_local_verify.log
  pass "verify_scripts"
else
  tail -n 20 /tmp/ci_local_verify.log
  fail "verify_scripts — fix failures before pushing"
  exit 1
fi

if [ "$BUILD" = "1" ]; then
  echo
  echo "== [3/3] docker build (no push) =="
  if docker build -f Dockerfile -t ci-local-check .; then
    pass "docker build"
  else
    fail "docker build"
    exit 1
  fi
fi

echo
pass "LOCAL CI GREEN — safe to push"
