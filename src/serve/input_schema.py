from typing import List, Dict
from pydantic import BaseModel
import json
import os


def load_sample_input():
    """Load a single-row example from sample_input.json for docs."""
    try:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        sample_path = os.path.join(project_root, "sample_input.json")
        with open(sample_path, "r") as f:
            data = json.load(f)
            return data[:1]
    except Exception as e:
        print(f"Failed to load sample input: {e}")
        return [{"error": "Failed to load"}]


class InputData(BaseModel):
    input: List[Dict[str, float]]
    model_config = {"json_schema_extra": {"examples": [{"input": load_sample_input()}]}}
