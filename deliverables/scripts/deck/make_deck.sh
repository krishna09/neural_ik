#!/bin/bash
# Build the deck end to end: assets -> pptxgenjs -> animations/transitions -> validation -> preview images.
#   NIK_WORK=/path/to/work PPTX_SKILL=/path/to/pptx/skill bash deliverables/scripts/deck/make_deck.sh [--no-assets]
# Needs: node with pptxgenjs 3.12 (npm install in deliverables/scripts/deck, or NODE_PATH), python with lxml,
# LibreOffice (soffice) and poppler (pdftoppm) for the preview images.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
DECK="$ROOT/deliverables/presentation/Neural_IK_STP_Team06_Level1.pptx"
QA="${NIK_WORK:-/tmp/nik_work}/deck/qa"
export NODE_PATH="${NODE_PATH:-$HERE/node_modules}"
if [ "${1:-}" != "--no-assets" ]; then python "$HERE/prepare_deck_assets.py"; fi
if [ -n "${PPTX_SKILL:-}" ]; then export APPLY_THEME="$PPTX_SKILL/scripts/apply_theme.js"; fi
node "$HERE/build_deck.js"
python "$HERE/postprocess_deck.py" "$DECK"
if [ -n "${PPTX_SKILL:-}" ]; then python "$PPTX_SKILL/scripts/office/validate.py" "$DECK"; fi
python "$HERE/export_notes.py" "$DECK"
mkdir -p "$QA" && rm -f "$QA"/slide-*.jpg "$QA"/deck.pdf
cp "$DECK" "$QA/deck.pptx"
if [ -n "${PPTX_SKILL:-}" ]; then (cd "$QA" && timeout 900 python "$PPTX_SKILL/scripts/office/soffice.py" --headless --convert-to pdf deck.pptx >/dev/null 2>&1 || true)
else (cd "$QA" && timeout 900 soffice --headless --convert-to pdf deck.pptx >/dev/null 2>&1 || true); fi
pdftoppm -jpeg -r 110 "$QA/deck.pdf" "$QA/slide" 2>/dev/null
ls "$QA"/slide-*.jpg | wc -l
