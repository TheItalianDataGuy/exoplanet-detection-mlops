from typing import List, Dict
from pydantic import BaseModel
import json
import os


def load_sample_input() -> List[Dict[str, float]]:
    """
    Load a single-row sample input from sample_input.json for Swagger documentation.

    Returns:
        A list containing a single dictionary of input features, or a fallback dict if loading fails.
    """
    try:
        # Navigate up to the project root
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        sample_path = os.path.join(project_root, "sample_input.json")

        # Load JSON and return only the first record
        with open(sample_path, "r") as f:
            data = json.load(f)
            return data[:1]
    except Exception as e:
        print(f"Failed to load sample input: {e}")
        return [{"error": "Failed to load"}]


class InputData(BaseModel):
    input: List[Dict[str, float]]

    # Inject example for use in OpenAPI docs (Swagger UI)
    model_config = {"json_schema_extra": {"examples": [{"input": load_sample_input()}]}}
