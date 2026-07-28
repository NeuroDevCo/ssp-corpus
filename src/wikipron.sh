#!/bin/bash

echo "Searching for $1..."

uv run wikipron $1 --syllable-boundaries --stress --narrow > data/raw/$1.tsv 