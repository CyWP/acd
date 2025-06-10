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
