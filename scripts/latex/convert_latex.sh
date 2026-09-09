#!/bin/bash

# Simple automated pipeline for LaTeX to HTML conversion
# Usage: ./convert_latex.sh main.tex

if [ $# -eq 0 ]; then
    echo "Usage: $0 <latex_file.tex>"
    exit 1
fi

INPUT_FILE="$1"
BASE_NAME="${INPUT_FILE%.tex}"
HTML_FILE="${BASE_NAME}.html"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Converting $INPUT_FILE to HTML..."

# Step 1: Use LaTeXML to convert LaTeX to HTML with proper cross-references
latexmlc "$INPUT_FILE" --destination="$SCRIPT_DIR/out/$HTML_FILE" --format=html5 --nocomments --log=/dev/null

if [ $? -ne 0 ]; then
    echo "Error: LaTeXML conversion failed"
    exit 1
fi

# Step 2: Run LaTeXML post-processing script
tsx "$SCRIPT_DIR/postprocess_latexml.ts" "$SCRIPT_DIR/out/$HTML_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Footnote fixing failed"
    exit 1
fi
