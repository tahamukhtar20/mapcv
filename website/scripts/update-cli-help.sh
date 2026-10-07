#!/usr/bin/env bash
# Regenerate the `--help` snapshots shown on the CLI reference page.
# Run with mapcv installed (for example after `uv run maturin develop`):
#   website/scripts/update-cli-help.sh                  # uses `mapcv` on PATH
#   MAPCV="uv run mapcv" website/scripts/update-cli-help.sh
set -euo pipefail
out="$(cd "$(dirname "$0")/.." && pwd)/src/cli-help"
read -r -a mapcv_cmd <<< "${MAPCV:-mapcv}"
mkdir -p "$out"

snapshot() {
  # Fixed width and no colour; drop trailing spaces and leading blank lines.
  COLUMNS=88 NO_COLOR=1 TERM=dumb "${mapcv_cmd[@]}" "$@" --help | sed 's/[ ]*$//' | sed '/./,$!d'
}

snapshot > "$out/main.txt"
for command in init plan generate info split stats card verify export validate cache mcp; do
  snapshot "$command" > "$out/$command.txt"
done
"${mapcv_cmd[@]}" --version > "$out/version.txt"
echo "Wrote CLI help snapshots to $out"
