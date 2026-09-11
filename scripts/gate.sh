#!/usr/bin/env bash
# ADDED BY SOURAV -- the single source of truth for this repo's
# Pre-Human-Review Quality Gate. Runs identically on a dev machine,
# inside a Claude Code hook, and in CI (blueprint 2.4/4.1) -- it is a
# thin wrapper because the actual work (and the honesty discipline
# around what is real vs. not_configured) lives in gate-report.py, in
# ONE place, not duplicated between this script and CI YAML.
#
#   bash scripts/gate.sh --fast   # PostToolUse hook: lint + tests + secret/PHI scans
#   bash scripts/gate.sh --full   # Stop hook / before requesting review: everything + gate-report.json/.md
#
# Exits nonzero if any MANDATORY check failed (see gate-report.py's
# MANDATORY_CHECKS) -- callers (hooks, CI) should treat nonzero as a red
# gate, per blueprint 4.2's Stop-hook example.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

MODE="full"
case "${1:-}" in
  --fast) MODE="fast" ;;
  --full|"") MODE="full" ;;
  *) echo "usage: gate.sh [--fast|--full]" >&2; exit 64 ;;
esac

PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON="python"

"$PYTHON" "$REPO_ROOT/scripts/gate-report.py" --mode "$MODE"
exit $?
