#!/usr/bin/env bash

set -e

file="$1"

if [[ -z "$file" ]]; then
    echo "Usage: $0 file.md"
    exit 1
fi

curl -s \
    -X POST \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    https://api.github.com/markdown \
    -d "$(jq -n \
        --arg text "$(cat "$file")" \
        '{text: $text, mode: "gfm"}')"
