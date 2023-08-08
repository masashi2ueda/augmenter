# %%
import torch
import torchvision.transforms as T


from typing import Any, List, Union


# %%
def apply_samet_func(func: Any, *srcs_chw: List[torch.tensor]) -> Union[List[torch.tensor], torch.Tensor]:
    """apply same transporse to tensors.

    Args:
        func (Any): transporse
        *srcs_chw (List[torch.tensor]): src_tensors

    Returns:
        Union[List[torch.tensor], torch.Tensor]: if src is one tensor -> return a tensor.

    Examples:
        >>> import torch
        >>> import torchvision.transforms as T
        >>> func = T.RandomHorizontalFlip(1)
        >>> img_chw = torch.zeros((3, 10, 10))
        >>> mask_chw = torch.zeros((1, 10, 10))
        >>> img_chw, mask_chw = apply_samet_func(func, img_chw, mask_chw)
        >>> print(img_chw.shape, mask_chw.shape)
        torch.Size([3, 10, 10]) torch.Size([1, 10, 10])
    """
    merged_chw = torch.concat(srcs_chw, dim=0)
    merged_chw = func(merged_chw)

    chs = [src_chw.shape[0] for src_chw in srcs_chw]
    dsts_chw = []
    start_c = 0
    for ch in chs:
        dsts_chw.append(merged_chw[start_c:(start_c+ch), ...])
        start_c = start_c + ch

    if len(dsts_chw) == 0:
        return dsts_chw[0]

    return dsts_chw


def apply_resize_func(src_img_chw: torch.tensor, resize_func: T.Resize, verbose: bool = False) -> torch.Tensor:
    """if same size not apply.

    Args:
        src_img_chw (torch.tensor): 
        resize_func (T.Resize): 
        verbose (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: 

    Examples:
        >>> import torch
        >>> import torchvision.transforms as T
        >>> resize_func = T.Resize((10, 20))
        >>> src_img_chw = torch.zeros((3, 10, 10))
        >>> dst_img_chw = apply_resize_func(src_img_chw, resize_func, True)
        resized
        >>> print(dst_img_chw.shape)
        torch.Size([3, 10, 20])
        >>> src_img_chw = torch.zeros((3, 10, 20))
        >>> dst_img_chw = apply_resize_func(src_img_chw, resize_func, True)
        not resized
        >>> print(dst_img_chw.shape)
        torch.Size([3, 10, 20])
    """
    _, src_h, src_w = src_img_chw.shape
    if src_h == resize_func.size[0] and src_w == resize_func.size[1]:
        if verbose:
            print("not resized")
        return src_img_chw
    if verbose:
        print("resized")
    return resize_func(src_img_chw)


# %%
