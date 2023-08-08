# %%
import torch
import torch.nn as nn
import ttach as tta
import matplotlib.pyplot as plt

from einops import rearrange


# %%
def create_tta_segwrapper(
    model,
    is_horizontal_flip: bool = True,
    is_vertical_flip: bool = True,
    is_rotate_90: bool = True,
    is_rotate_180: bool = True,
    is_rotate_270: bool = True,
    is_multiply: bool = True,
    merge_mode: str = "mean"
) -> tta.SegmentationTTAWrapper:
    """wrapper of tta

    Args:
        model (_type_): _description_
        is_horizontal_flip (bool, optional): _description_. Defaults to True.
        is_vertical_flip (bool, optional): _description_. Defaults to True.
        is_rotate_90 (bool, optional): _description_. Defaults to True.
        is_rotate_180 (bool, optional): _description_. Defaults to True.
        is_rotate_270 (bool, optional): _description_. Defaults to True.
        is_multiply (bool, optional): _description_. Defaults to True.
        merge_mode (str, optional): _description_. Defaults to "mean".

    Returns:
        tta.SegmentationTTAWrapper: _description_

    Examples:
        >>> class CustomModel(nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>     def forward(self, inputs_bchw):
        >>>         inputs_bchw[:, :, :10, :20] = 0.25
        >>>         return inputs_bchw
        >>> model = CustomModel()
        >>> tta_model = create_ttamodel(model)
        >>> img_bchw = torch.zeros((1, 3, 50, 50))
        >>> res_bchw = tta_model(img_bchw)
        >>> myshow = lambda x: plt.imshow(rearrange(x, "c h w -> h w c"))
        >>> myshow(res_bchw[0])
    """
    transforms = []
    if is_horizontal_flip:
        transforms.append(tta.HorizontalFlip())
    if is_vertical_flip:
        transforms.append(tta.VerticalFlip())
    rots = []
    if is_rotate_90:
        rots.append(90)
    if is_rotate_180:
        rots.append(180)
    if is_rotate_270:
        rots.append(270)
    if len(rots) != 0:
        transforms.append(tta.Rotate90(angles=rots))

    if is_multiply:
        transforms.append(tta.Multiply(factors=[0.9, 1, 1.1]))

    transforms = tta.Compose(transforms)
    tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode=merge_mode)
    return tta_model

# %%