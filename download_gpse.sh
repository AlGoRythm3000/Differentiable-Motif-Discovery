#!/usr/bin/env bash
# Download a pretrained GPSE encoder into GPSE_pretrained/.
#
# The weights (~254 MiB each) are not tracked by git; run this script once
# after cloning. Source: Zenodo record 8145095, released with
# "Graph Positional and Structural Encoder" (Cantürk et al., ICML 2024).
#
# Usage:
#   ./download_gpse.sh            # default: molpcba (the one used in this repo)
#   ./download_gpse.sh zinc       # or: pcqm4mv2, geom, chembl
#   ./download_gpse.sh all        # every variant

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$REPO_DIR/GPSE_pretrained"
BASE_URL="https://zenodo.org/records/8145095/files"

# variant -> sha256 of gpse_model_<variant>_1.0.pt
declare -A SHA256=(
  [molpcba]="7bc81da2231cc1a6898683317fe0a278ffa4953540bbad375e2c20af840f9cdd"
)

VARIANTS_ALL=(molpcba zinc pcqm4mv2 geom chembl)

variant="${1:-molpcba}"
if [[ "$variant" == "all" ]]; then
  variants=("${VARIANTS_ALL[@]}")
elif [[ " ${VARIANTS_ALL[*]} " == *" $variant "* ]]; then
  variants=("$variant")
else
  echo "Unknown variant '$variant'. Choose one of: ${VARIANTS_ALL[*]} (or 'all')." >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

for v in "${variants[@]}"; do
  file="gpse_model_${v}_1.0.pt"
  dest="$DEST_DIR/$file"

  if [[ -f "$dest" ]]; then
    echo "==> $file already present, skipping download."
  else
    echo "==> Downloading $file (~254 MiB) into GPSE_pretrained/ ..."
    if command -v curl >/dev/null 2>&1; then
      # -C - resumes a partial download if the script was interrupted.
      curl -L -C - --fail --progress-bar -o "$dest" "$BASE_URL/$file?download=1"
    elif command -v wget >/dev/null 2>&1; then
      wget -c -O "$dest" "$BASE_URL/$file?download=1"
    else
      echo "Neither curl nor wget is available. Install one of them, or download" >&2
      echo "$BASE_URL/$file manually into GPSE_pretrained/." >&2
      exit 1
    fi
  fi

  if [[ -n "${SHA256[$v]:-}" ]] && command -v sha256sum >/dev/null 2>&1; then
    echo "==> Verifying checksum ..."
    echo "${SHA256[$v]}  $dest" | sha256sum --check --status \
      && echo "    ok: $file" \
      || { echo "    CHECKSUM MISMATCH for $file - delete it and retry." >&2; exit 1; }
  fi
done

echo "Done. Weights are in GPSE_pretrained/."
