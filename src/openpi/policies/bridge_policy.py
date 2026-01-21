import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_bridge_example() -> dict:
    """Creates a random input example for the Bridge policy."""
    return {
        "state": np.random.rand(8),
        "images": {
            "image_0": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
            "image_1": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
            "image_2": np.random.randint(256, size=(256, 256, 3), dtype=np.uint8),
        },
        "actions": np.random.rand(7),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class BridgeInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # parse image
        base_image = _parse_image(data["images"]["image_0"])
        left_image = _parse_image(data["images"]["image_1"])
        right_image = _parse_image(data["images"]["image_2"])

        # check if side cameras are available
        if left_image.mean() == 0.0:
            side_cameras = False
        else:
            side_cameras = True
        
        if side_cameras:
            image_masks = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            }
        else:
            image_masks = {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            }
        
        inputs = {
            "state": data["state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": image_masks,
        }

        # actions

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class BridgeOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}