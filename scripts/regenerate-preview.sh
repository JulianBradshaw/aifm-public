#!/usr/bin/env bash
# Regenerate public/preview.jpg (the Open Graph / Twitter card image) from a
# fresh hi-DPI screenshot of https://www.aifuturesmodel.com/.
#
# The OG metadata in app/layout.tsx declares 1200x630. We capture at 1710x1100
# logical viewport with a 2x device scale factor (3420x2200 source) so text
# stays crisp after downsampling, then crop the top 1200x630 so the rendered
# preview includes the title row, all chart panels, the key parameters row,
# the "The AI Futures Model" heading, and the model diagram — without trailing
# whitespace from the next section.
#
# Requirements: chromium, imagemagick (magick).
# Usage: scripts/regenerate-preview.sh [url]

set -euo pipefail

URL="${1:-https://www.aifuturesmodel.com/}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO_ROOT/public/preview.jpg"
TMP_PNG="$(mktemp --suffix=.png)"
trap 'rm -f "$TMP_PNG"' EXIT

chromium \
  --headless \
  --disable-gpu \
  --hide-scrollbars \
  --window-size=1710,1100 \
  --force-device-scale-factor=2 \
  --virtual-time-budget=10000 \
  --screenshot="$TMP_PNG" \
  "$URL"

magick "$TMP_PNG" \
  -resize 1200x \
  -gravity north -crop 1200x630+0+0 +repage \
  -quality 92 \
  "$OUT"

echo "Wrote $OUT ($(identify -format '%wx%h, %b' "$OUT"))"
