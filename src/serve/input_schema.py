from typing import List, Dict
from pydantic import BaseModel, Field
from pathlib import Path
import json
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)


def _load_sample_input() -> List[Dict[str, float]]:
    """
    Loads a sample input from a local JSON file to populate the FastAPI Swagger example.

    Returns:
        A list containing a single input record (dict of features), or a fallback example if loading fails.
    """
    try:
        # Resolve the path to the sample_input.json file located two levels up from this file
        sample_path = Path(__file__).resolve().parents[2] / "sample_input.json"
        logger.info(f"📂 Attempting to load sample input from: {sample_path}")

        # Load and validate the JSON structure
        with open(sample_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                return data[
                    :1
                ]  # Use only the first record to avoid clutter in Swagger docs
            else:
                logger.warning("sample_input.json exists but is not a non-empty list.")
    except Exception as e:
        logger.warning(f"Failed to load sample_input.json: {e}")

    # Fallback example structure (minimal valid input)
    return [{"koi_period": 1.0}]


class InputData(BaseModel):
    """
    Schema for the input payload to the /predict endpoint.

    Attributes:
        input (List[Dict[str, float]]): A list of dictionaries, each representing a single feature vector.
    """

    input: List[Dict[str, float]] = Field(..., example=_load_sample_input())
