from dataclasses import asdict

import torch

from deepwarp.model import DeepWarp, Layout
from deepwarp.settings import model_settings_from


def save_model(model, path):
    torch.save(
        {
            "layout": asdict(model.layout),
            "settings": asdict(model.settings),
            "description": model.description,
            "weights": {name: value.cpu() for name, value in model.state_dict().items()},
        },
        path,
    )


def load_model(path, device="cpu"):
    stored = torch.load(path, map_location=device, weights_only=True)
    model = DeepWarp(Layout(**stored["layout"]), model_settings_from(stored["settings"]))
    model.load_state_dict(stored["weights"])
    model.description = stored["description"]
    return model.to(device).eval()
