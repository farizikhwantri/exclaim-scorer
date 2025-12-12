#!/bin/bash

# input from argument or default output path
input_json="${1:-/Users/fariz/repositories/exclaim-scorer/config/example_simple.json}"
output_json="${2:-/Users/fariz/repositories/exclaim-scorer/temp/output_simple.json}"

curl -s -X POST http://localhost:8000/score \ 
  -H "Content-Type: application/json" \
  -d @- < "$input_json" > "$output_json"

echo "Scoring completed. Output saved to $output_json"
cat "$output_json"
