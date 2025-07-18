from typing import List, Dict
from pydantic import BaseModel, Field


class InputData(BaseModel):
    input: List[Dict[str, float]] = Field(
        ...,  # Required
        example=[
            {
                "koi_fpflag_nt": 0,
                "koi_fpflag_ss": 0,
                "koi_fpflag_co": 0,
                "koi_fpflag_ec": 0,
                "koi_period": 41.07962,
                "koi_time0bk": 133.5268,
                "koi_impact": 0.25,
                "koi_duration": 3.953,
                "koi_depth": 693.1,
                "koi_prad": 2.09,
                "koi_teq": 755,
                "koi_insol": 211,
                "koi_model_snr": 13.5,
                "koi_tce_plnt_num": 1,
                "koi_steff": 5703,
                "koi_slogg": 4.47,
                "koi_srad": 0.89,
                "ra": 294.3,
                "dec": 46.0,
                "koi_kepmag": 15.3,
            }
        ],
    )
