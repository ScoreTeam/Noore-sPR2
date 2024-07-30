# built-in dependencies
from typing import Any

# project dependencies
from FaceRec.NFR2.deepface.basemodels import (
    NFRmodel,
)

def build_model(model_name: str) -> Any:
    global model_obj

    models = {
        "NFR": NFRmodel.ArcFaceClient,
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model_obj[model_name] = model()
        else:
            raise ValueError(f"Invalid model_name passed - {model_name}")
    return model_obj[model_name]
