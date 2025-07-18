import json

# Paths
SAMPLE_INPUT_PATH = "sample_input.json"
OUTPUT_PATH = "models/expected_columns.json"

# Load sample input
with open(SAMPLE_INPUT_PATH, "r") as f:
    sample_input = json.load(f)

# Extract column names
expected_columns = list(sample_input[0].keys())

# Save to file
with open(OUTPUT_PATH, "w") as f:
    json.dump(expected_columns, f, indent=4)

print(f"expected_columns.json written to {OUTPUT_PATH}")
