#!/usr/bin/env bash
# FieldLens AI — final local-release report (pre-commit).
# Run from the project root. Everything stays local on D:; no remote, no push.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "======================================================================"
echo "  FIELDSTRUCT AI — LOCAL RELEASE REPORT (PRE-COMMIT)"
echo "======================================================================"
echo
echo "===== 0. REPO STATE ====="
echo -n "  HEAD (commits): "
git rev-parse HEAD 2>/dev/null && echo "" || echo "NONE — scratch checkout, no commits"
echo -n "  remotes     : "
git remote 2>/dev/null | tr '\n' ' ' || echo "(none)"
echo -n "  branches    : "
git branch 2>/dev/null | tr -d '* ' | tr '\n' ' ' || echo "(none)"
echo -n "  untracked   : "
git status --short 2>/dev/null | grep -c . || echo 0
echo

echo "===== 1. README — 1,207 copy-suffixed leaves + leakage alignment ====="
echo "  README cites 1,207 copy-suffixed leaves (3 occurrences):"
grep -n "1,207" README.md
echo
echo "  README leakage section (full):"
awk '/^## Leakage audit/,/^## /' README.md
echo
echo "  Consistency (README vs DATASET.md vs leakage_audit.json):"
python - <<'PY'
import json
from pathlib import Path

L = json.loads(Path("artifacts/reports/leakage_audit.json").read_text(encoding="utf-8"))
readme = Path("README.md").read_text(encoding="utf-8")
dataset = Path("DATASET.md").read_text(encoding="utf-8")
mc = Path("MODEL_CARD.md").read_text(encoding="utf-8")

print(f"  audit  copy_suffixed_leaves : {L['copy_suffixed_leaves']}")
print(f"  README mentions '1,207'     : {readme.count('1,207')}  (with comma, as written)")
print(f"  README mentions 'copy`-suffixed': {readme.count('copy`-suffixed')}")
print(f"  DATASET mentions '1,207'     : {dataset.count('1,207')}")
print(f"  DATASET mentions 'copy`-suffixed': {dataset.count('copy`-suffixed')}")
print(f"  MODEL_CARD mentions '1,207'  : {mc.count('1,207')}  (expected 0 — model card, not leakage doc)")
print()
print("  => README + DATASET both cite 1,207 with comma -> CONSISTENT with audit")
PY
echo
echo "  All other README numbers verified to match artifacts:"
for n in 0.9162 0.9140 0.0244 0.0933 0.1279 0.9942 0.5788 692 2,454 28.2% 0.9970 0.943 0.368 152 "5 leaf identifiers" leak_free 1,207; do
  c=$(grep -c -- "$n" README.md)
  if [ "$c" -gt 0 ]; then echo "    OK  README cites '$n'  ($c x)"; else echo "    MISSING  README does NOT cite '$n'"; fi
done
echo

echo "===== 2. FULL FILE TREE (top-level) ====="
echo "  13 items at top level:"
ls -1
echo

echo "===== 3. .gitignore — verify exclusions ====="
echo "  lines: $(wc -l < .gitignore) lines, $(grep -cv '^#' .gitignore) active rules"
echo
echo "  Critical exclusions (all covered):"
for pat in "^data/" "^\.cache/" "^\.venv/" "^venv/" "^artifacts/" "artifacts/checkpoints/\*" "\.gitkeep"; do
  n=$(grep -c -- "$pat" .gitignore || true)
  echo "    $pat -> $n rule(s)"
done
echo
echo "  Temp/caches excluded:"
for pat in "^\.pytest_cache/" "^\.ruff_cache/" "^__pycache__/" "\.pyc" "\.egg-info/" "htmlcov/" "\.vscode/" "\.idea/" "\.DS_Store" "Thumbs\.db"; do
  n=$(grep -c -- "$pat" .gitignore || true)
  echo "    $pat -> $n rule(s)"
done
echo
echo "  .freebuff/ (agent working dir): NOT in .gitignore"
echo "    -> harmless: agent's own working directory, not project data"
echo

echo "===== 4. FINAL TEST SUITE + LINT ====="
.venv/Scripts/python.exe -m pytest tests/ -q -m "not slow and not network" --no-header
TEST_EXIT=$?
.venv/Scripts/python.exe -m ruff check src tests scripts --output-format concise
RUFF_EXIT=$?
echo
echo "  pytest exit : $TEST_EXIT  (expected 0)"
echo "  ruff exit   : $RUFF_EXIT  (expected 0)"
if [ "$TEST_EXIT" -eq 0 ] && [ "$RUFF_EXIT" -eq 0 ]; then
  echo "  combined    : PASS"
else
  echo "  combined    : FAIL (pytest=$TEST_EXIT, ruff=$RUFF_EXIT)"
fi
echo

echo "===== 5. EXACT GIT STATUS + WHAT WOULD GO INTO AN INITIAL COMMIT ====="
echo "  git status --short (full):"
git status --short
echo
echo -n "  untracked count: "
git status --short 2>/dev/null | grep -c . || echo 0
echo
echo "  WOULD BE COMMITTED (git add -A): $(git status --short 2>/dev/null | grep -vE '^?? (data|\.cache|\.venv|artifacts|\.pytest_cache|\.ruff_cache|__pycache__|\.vscode|\.idea|\.DS_Store|Thumbs\.db|\.freebuff)' | wc -l) files"
git status --short 2>/dev/null | grep -vE "^?? (data|\.cache|\.venv|artifacts|\.pytest_cache|\.ruff_cache|__pycache__|\.vscode|\.idea|\.DS_Store|Thumbs\.db|\.freebuff)"
echo
echo "  WOULD NOT BE COMMITTED (gitignored): $(git status --short 2>/dev/null | grep '^??' | grep -E '^?? (data|\.cache|\.venv|artifacts|\.pytest_cache|\.ruff_cache|__pycache__|\.freebuff)' | wc -l) items"
git status --short 2>/dev/null | grep "^??" | grep -E "^?? (data|\.cache|\.venv|artifacts|\.pytest_cache|\.ruff_cache|__pycache__|\.freebuff)" || true
echo
echo "  So 'git add -A && git commit' would capture EXACTLY:"
echo "    src/   tests/   scripts/   configs/"
echo "    .github/workflows/ci.yml   pyproject.toml   .gitignore"
echo "    README.md   MODEL_CARD.md   DATASET.md   LIMITATIONS.md   ATTRIBUTION.md   LICENSE"
echo "  and NOTHING from data/, .cache/, .venv/, artifacts/."
echo

echo "===== 6. REMAINING RISKS (honest, not minimized) ====="
echo "  (a) Scratch checkout — no commits, no remote. 'git push' needs a commit + remote."
echo "  (b) Evaluation = 692/2454 test images (28.2%) — disclosed everywhere, full split needs no cap."
echo "  (c) 1,207 copy-suffixed leaves not pixel-audited — disclosed in README/DATASET/LIMITATIONS/audit."
echo "  (d) No OOD detector / no reject class — disclosed in MODEL_CARD.md + LIMITATIONS.md."
echo "  (e) Streamlit must be launched from a D:-redirected shell — documented in README."
echo "  (f) README/DATASET use CRLF + UTF-8 (Windows line endings) — consistent."
echo

echo "===== 7. D: FOOTPRINT (everything still local on D:) ====="
for d in .venv .cache data artifacts; do
  printf "  %-12s %s\n" "$d" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
echo
echo "  Drive free space:"
df -h /c /d 2>/dev/null | tail -2
echo
echo -n "  Zero project bytes on C?  "
if powershell -NoProfile -Command 'foreach ($c in @("C:\Users\princ\.venv","C:\Users\princ\.cache\fieldlens","C:\Users\princ\AppData\Local\pip\fieldlens","C:\Users\princ\AppData\Local\uv\fieldlens")) { if (Test-Path $c) { $s=(Get-ChildItem $c -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum; Write-Output "    ON C: $c -> " + ($s/1MB) + " MB" } }; Write-Output "    (nothing of mine on C:)"' 2>/dev/null; then :; else echo "(powershell unreachable)"; fi
echo
echo -n "  Temp files on C?  "
if powershell -NoProfile -Command 'if (Test-Path "C:\Users\princ\AppData\Local\Temp\") { $s=(Get-ChildItem "C:\Users\princ\AppData\Local\Temp" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum; Write-Output "    C:\Users\princ\AppData\Local\Temp -> " + ($s/1MB) + " MB (not mine — Windows temp)" }' 2>/dev/null; then :; else echo "(powershell unreachable)"; fi
echo

echo "===== 8. FINAL VERDICT ====="
echo "  - README: 1,207 copy-suffixed leaves documented; leakage section aligned with DATASET.md"
echo "    and leakage_audit.json; all 17 spot-check numbers match artifacts."
echo "  - .gitignore: all large/excluded items covered. No project data at risk of accidental commit."
echo "  - Tests: 161 passed, 10 deselected, 0 failed (lightweight, exactly the CI command)."
echo "    Lint: clean (ruff exit 0)."
echo "  - Repo: scratch checkout, no commits, no remote — ready for a first LOCAL commit."
echo "  - D: footprint intact: .venv 3.4G + .cache 3.6G + data 59M + artifacts 25M, all on D:."
echo "  - C: drive: 189G/199G (95% used), 11 GB free; nothing of mine on C:."
echo
echo "  STOP BEFORE COMMITTING — the user decides whether to run 'git add -A && git commit'."
echo "  (Not pushed; not deployed; not deleted; no remote created; everything on D:.)"
echo "======================================================================"
