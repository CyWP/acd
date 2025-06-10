from PIL import Image
import torch
from torchvision import transforms


def pil_to_tensor(img):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to tensor with values in range [0, 1]
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            ),  # Normalize to [-1, 1]
        ]
    )
    return transform(img)


def tensor_to_pil(tensor):
    tensor = (tensor * 0.5) + 0.5
    # Convert to numpy array and ensure values are in range [0, 255]
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    # Convert to PIL image
    return Image.fromarray(array)


def tensors_to_pil(tensor):
    if tensor.ndim == 3:
        return [tensor_to_pil(tensor)]
    if tensor.ndim == 4:
        return [tensor_to_pil(tensor[i]) for i in range(tensor.shape[0])]
    raise ValueError(
        f"Tensor must have 3 or 4 dimensions.\n Tensor shape: {tensor.shape}, dimensions: {tensor.ndim}"
    )


def bw_to_rgb(img):
    if img.ndim == 3:
        return img.repeat(3, 1, 1)
    return img.repeat(1, 3, 1, 1)


def tile_images(x, w, h):
    assert x.ndim == 4  # NCHW => CHW
    return (
        x.reshape(h, w, *x.shape[1:])
        .permute(2, 0, 3, 1, 4)
        .reshape(x.shape[1], h * x.shape[2], w * x.shape[3])
    )


def one_hot(idx, size, dtype=torch.float32):
    tensor = torch.zeros(size, dtype=dtype)
    tensor[idx] = 1.0
    return tensor


def one_hot_to_idx(tensor):
    return list(torch.nonzero(tensor, as_tuple=True)[-1].to(torch.device("cpu")))


def extract_center_img(image: torch.Tensor, size: int) -> torch.Tensor:
    """
    Extracts a center square patch of given size from a tensor image or batch.

    Parameters:
    - image: torch.Tensor of shape (H, W), (C, H, W), or (N, C, H, W)
    - size: int, desired size of the center square patch (size x size)

    Returns:
    - torch.Tensor with the same leading dimensions and (size, size) spatial shape
    """
    if image.ndim == 2:
        H, W = image.shape
        y, x = (H - size) // 2, (W - size) // 2
        return image[y : y + size, x : x + size]

    elif image.ndim == 3:
        C, H, W = image.shape
        y, x = (H - size) // 2, (W - size) // 2
        return image[:, y : y + size, x : x + size]

    elif image.ndim == 4:
        N, C, H, W = image.shape
        y, x = (H - size) // 2, (W - size) // 2
        return image[:, :, y : y + size, x : x + size]

    else:
        raise ValueError("Input must be 2D, 3D, or 4D tensor.")


import torch
import torch.nn.functional as F


def resize_tensor_img(
    image: torch.Tensor, size, mode="bilinear", align_corners=False
) -> torch.Tensor:
    """
    Resizes a tensor image or batch of images to the specified size.

    Parameters:
    - image: torch.Tensor of shape (H, W), (C, H, W), or (N, C, H, W)
    - size: int or tuple (H_out, W_out)
    - mode: str, interpolation mode (e.g., 'bilinear', 'nearest', 'bicubic', 'area')
    - align_corners: bool, only relevant for 'bilinear' and 'bicubic'

    Returns:
    - Resized torch.Tensor
    """
    if isinstance(size, int):
        size = (size, size)

    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        resized = F.interpolate(
            image, size=size, mode=mode, align_corners=align_corners
        )
        return resized.squeeze(0).squeeze(0)

    elif image.ndim == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)
        resized = F.interpolate(
            image, size=size, mode=mode, align_corners=align_corners
        )
        return resized.squeeze(0)

    elif image.ndim == 4:
        return F.interpolate(image, size=size, mode=mode, align_corners=align_corners)

    else:
        raise ValueError("Input tensor must be 2D, 3D, or 4D.")
