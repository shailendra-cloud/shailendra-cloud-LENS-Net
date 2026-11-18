#!/usr/bin/env bash
# Download and unpack the per-window predictions of every model (97 MB, 824 MB unpacked).
#   bash reproduce/get_predictions.sh                         # download the release asset
#   bash reproduce/get_predictions.sh /path/to/lensnet_predictions.tar.xz   # use a local copy
# Unpacks into workspace/ and source_benchmark/ at the repository root, where every analysis script looks.
set -euo pipefail
cd "$(dirname "$0")/.."

ASSET="releases/download/v1.0/lensnet_predictions.tar.xz"
URLS="https://github.com/shailendra-cloud/LENS-Net/$ASSET https://github.com/Soma2024-08/LENS-Net/$ASSET"     # the second is the first author's copy of the same file
SHA256="b48a455266922266ca1109c9bd97c693332578831feba0485fc3b31d8bc1582d"
FILE="${1:-lensnet_predictions.tar.xz}"

if [ ! -f "$FILE" ]; then
  for URL in $URLS; do
    echo "Downloading $URL"
    if curl -L --fail -o "$FILE" "$URL"; then break; fi
    rm -f "$FILE"
  done
  [ -f "$FILE" ] || { echo "Download failed. Get lensnet_predictions.tar.xz from the release page and pass its path to this script."; exit 1; }
fi

echo "Checking SHA-256 ..."
echo "$SHA256  $FILE" | sha256sum -c -

echo "Unpacking ..."
tar -xJf "$FILE"

# The result summaries live in results/ only: the analysis scripts read them through workspace/.
ln -sfn ../results workspace/logs
for f in results/revision/*; do ln -sfn "../../$f" "workspace/review/$(basename "$f")"; done
echo "Done. Predictions are in workspace/ and source_benchmark/. Next: python reproduce/verify_paper.py"
