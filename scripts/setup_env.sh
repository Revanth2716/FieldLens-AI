#!/usr/bin/env bash
# FieldLens AI - environment setup.
#
# On the target machine the system drive C: runs very close to full, so EVERY
# cache, temporary directory and the virtual environment is kept on the project
# drive. Without this, installing torch (~3 GB) plus its caches can exhaust C:.
#
# Usage (from Git Bash on Windows, or any POSIX shell):
#     source scripts/setup_env.sh          # export into the current shell
#     bash scripts/setup_env.sh --check    # report where caches would land
#
# The script locates itself, so it works from any working directory and does not
# hard-code a drive letter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Convert a POSIX path to a Windows-friendly path when running under Git Bash,
# so tools like uv and pip (native Windows binaries) understand it.
to_native() {
  if command -v cygpath >/dev/null 2>&1; then
    cygpath -m "$1"
  else
    printf '%s' "$1"
  fi
}

NATIVE_ROOT="$(to_native "${PROJECT_ROOT}")"
CACHE_ROOT="${NATIVE_ROOT}/.cache"

# Every location tools default to under C: is redirected here.
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
export UV_PYTHON_INSTALL_DIR="${CACHE_ROOT}/uv/python"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export TORCH_HOME="${CACHE_ROOT}/torch"
export MPLCONFIGDIR="${CACHE_ROOT}/matplotlib"
export XDG_CACHE_HOME="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export TMP="${CACHE_ROOT}/tmp"
export TEMP="${CACHE_ROOT}/tmp"
export TMPDIR="${CACHE_ROOT}/tmp"

mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TORCH_HOME}" \
         "${MPLCONFIGDIR}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TMP}"

report() {
  echo "FieldLens AI - environment"
  echo "  project root : ${PROJECT_ROOT}"
  echo "  cache root   : ${CACHE_ROOT}"
  for name in UV_CACHE_DIR PIP_CACHE_DIR TORCH_HOME HF_HOME TMP TEMP TMPDIR; do
    printf '  %-14s %s\n' "${name}" "$(printenv "${name}")"
  done
  if command -v df >/dev/null 2>&1; then
    echo "  project drive: $(df -h "${PROJECT_ROOT}" | tail -1 | awk '{print $4" free"}')"
    if [ -d /c ]; then
      echo "  system drive : $(df -h /c | tail -1 | awk '{print $4" free"}')"
    fi
  fi
}

if [ "${1:-}" = "--check" ]; then
  report
  exit 0
fi

report
echo
echo "Environment exported. Next:"
echo "  uv venv --python 3.12 .venv"
echo "  uv pip install --python .venv torch torchvision \\"
echo "      --index-url https://download.pytorch.org/whl/cu130"
echo "  uv pip install --python .venv -e '.[eval,ui,dev]'"
echo
echo "To make this persistent for every future shell on this machine, run:"
echo "  setx UV_CACHE_DIR \"${UV_CACHE_DIR}\""
echo "  setx PIP_CACHE_DIR \"${PIP_CACHE_DIR}\""
echo "  setx TORCH_HOME   \"${TORCH_HOME}\""
echo "  setx HF_HOME      \"${HF_HOME}\""
echo "  setx TMP          \"${TMP}\""
echo "  setx TEMP         \"${TEMP}\""
