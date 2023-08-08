# %%
from typing import Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from einops import rearrange

if __name__ == "__main__":
    # import sys
    # sys.path.append("..")
    import utils
    from mycrop import MyCrop
else:
    from . import utils
    from .mycrop import MyCrop


# %%
class Augmenter:
    def __init__(
            self,
            org_img_p: float = 0.5,
            is_horizontal_flip: bool = True,
            is_vetical_flip: bool = True,
            is_rotate_90: bool = True,
            is_rotate_180: bool = True,
            is_rotate_270: bool = True,
            is_crop: bool = True,
            crop_scale_min: float = 0.5,
            crop_scale_max: float = 1.0,
            is_multiple: float = False
    ) -> None:
        """_summary_

        Args:
            org_img_p (float, optional): _description_. Defaults to 0.5.
            is_horizontal_flip (bool, optional): _description_. Defaults to True.
            is_vetical_flip (bool, optional): _description_. Defaults to True.
            is_rotate_90 (bool, optional): _description_. Defaults to True.
            is_rotate_180 (bool, optional): _description_. Defaults to True.
            is_rotate_270 (bool, optional): _description_. Defaults to True.
            is_crop (bool, optional): _description_. Defaults to True.
            crop_scale_min (float, optional): _description_. Defaults to 0.5.
            crop_scale_max (float, optional): _description_. Defaults to 1.0.
            is_multiple (float, optional): _description_. Defaults to False.

        Examples:
            >>> w = 50
            >>> h = 50
            >>> img_chw = torch.zeros((3, h, w))
            >>> mask_chw = torch.zeros((1, h, w))
            >>> img_chw[0, 10:20, :30] = 1
            >>> mask_chw[0, 10:20, :30] = 1
            >>> augmenter = Augmenter(
            >>>     org_img_p=0.5,
            >>>     is_horizontal_flip=True,
            >>>     is_vetical_flip=True,
            >>>     is_rotate_90=True,
            >>>     is_rotate_180=True,
            >>>     is_rotate_270=True,
            >>>     is_crop=True,
            >>>     crop_scale_min=0.5,
            >>>     crop_scale_max=1.0,
            >>>     is_multiple = True)
            >>> img_chw2, mask_chw2 = augmenter.transform(False, img_chw, mask_chw)
            >>> print(img_chw2.shape, mask_chw2.shape)
            >>> myshow = lambda x: plt.imshow(rearrange(x, "c h w -> h w c"))
            >>> plt.figure(figsize=(3, 5))
            >>> plt.subplot(1, 2, 1)
            >>> myshow(img_chw)
            >>> plt.subplot(1, 2, 2)
            >>> myshow(img_chw2)
            >>> plt.pause(0.1)
            >>> plt.figure(figsize=(3, 5))
            >>> plt.subplot(1, 2, 1)
            >>> myshow(mask_chw)
            >>> plt.subplot(1, 2, 2)
            >>> myshow(mask_chw2)
            >>> plt.pause(0.1)
        """
        self.org_img_p = org_img_p
        transform_dict = {}
        if is_horizontal_flip:
            transform_dict["RandomHorizontalFlip"] = T.RandomHorizontalFlip(1)
        if is_vetical_flip:
            transform_dict["RandomVerticalFlip"] = T.RandomVerticalFlip(1)
        if is_rotate_90:
            transform_dict["RandomRotation90"] = T.RandomRotation(degrees=[90, 90])
        if is_rotate_180:
            transform_dict["RandomRotation180"] = T.RandomRotation(degrees=[180, 180])
        if is_rotate_270:
            transform_dict["RandomRotation270"] = T.RandomRotation(degrees=[270, 270])
        if is_crop:
            transform_dict["RandomCrop"] = MyCrop(crop_scale_min, crop_scale_max)
        self.transform_dict = transform_dict

        self.is_multiple = is_multiple

    def transform(self, verbose, *x_chw):

        if torch.rand(1)[0] < self.org_img_p:
            if verbose:
                print("out put org image")
            return x_chw
        
        keys = list(self.transform_dict.keys())
        if self.is_multiple:
            size = np.random.randint(1, len(keys))
        else:
            size = 1

        use_keys = np.random.choice(keys, size, replace=False)
        for key in use_keys:
            if verbose:
                print(key)
            x_chw = utils.apply_samet_func(self.transform_dict[key], *x_chw)
        return x_chw

# %%