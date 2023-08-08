# %%
import numpy as np
import torch


# %%
class MyCrop:
    def __init__(self, crop_scale_min: float, crop_scale_max: float) -> None:        
        self.crop_scale_min = crop_scale_min
        self.crop_scale_max = crop_scale_max

    def __call__(self, src_chw: torch.tensor, verbose: bool = False) -> torch.tensor:
        _, h, w = src_chw.shape
        scale = self.crop_scale_min + np.random.random()*(self.crop_scale_max - self.crop_scale_min)

        def resize(org):
            use = int(org * scale)
            s = np.random.randint(0, org - use)
            e = s + use
            return s, e
        sh, eh = resize(h)
        sw, ew = resize(w)
        dst_chw = src_chw[:, sh:eh, sw:ew]
        if verbose:
            print(f"scale:{scale}, sh:{sh}, eh:{eh}, sw:{sw}, ew:{ew}, dst_chw.shape:{dst_chw.shape}")

        return dst_chw
