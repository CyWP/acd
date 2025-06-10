from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F


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
    if tensor.shape[0] == 1:
        tensor = torch.cat([tensor, tensor, tensor], dim=0)
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


def is_one_hot(tensor: torch.Tensor) -> torch.Tensor:
    ones_count = (tensor == 1).sum(dim=-1)
    zeros_count = (tensor == 0).sum(dim=-1)
    length = tensor.size(-1)
    return (ones_count == 1) & (zeros_count == length - 1)


def is_one_hot_all(tensor: torch.Tensor) -> torch.Tensor:
    return is_one_hot(tensor).all().item()


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


def flip_horizontal(img: torch.Tensor) -> torch.Tensor:
    """
    Flips the image(s) horizontally (left-right).

    Supports shapes:
    - (C, H, W)
    - (N, C, H, W)

    Returns the horizontally flipped image(s).
    """
    if img.ndim == 3:
        return torch.flip(img, dims=[2])  # flip width axis
    elif img.ndim == 4:
        return torch.flip(img, dims=[3])  # flip width axis in batch
    else:
        raise ValueError("Input must be a 3D or 4D tensor (C,H,W) or (N,C,H,W)")


def flip_vertical(img: torch.Tensor) -> torch.Tensor:
    """
    Flips the image(s) vertically (up-down).

    Supports shapes:
    - (C, H, W)
    - (N, C, H, W)

    Returns the vertically flipped image(s).
    """
    if img.ndim == 3:
        return torch.flip(img, dims=[1])  # flip height axis
    elif img.ndim == 4:
        return torch.flip(img, dims=[2])  # flip height axis in batch
    else:
        raise ValueError("Input must be a 3D or 4D tensor (C,H,W) or (N,C,H,W)")


def extract_patches(img, h, w, as_grid=False):
    # Assert input is a torch.Tensor and has 3 or 4 dimensions
    assert isinstance(img, torch.Tensor), "Input must be a torch.Tensor"
    assert img.ndim == 4, f"Expected 3D or 4D tensor, got shape {img.shape}"

    B, C, H, W = img.shape

    # Check patch size validity
    assert h > 0 and w > 0, f"Patch size must be positive, got h={h}, w={w}"
    assert H % h == 0, f"Image height {H} must be divisible by patch height {h}"
    assert W % w == 0, f"Image width {W} must be divisible by patch width {w}"

    # Extract and permute patches
    patches = img.unfold(2, h, h).unfold(3, w, w)  # [B, C, H//h, W//w, h, w]
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, H//h, W//w, C, h, w]

    if as_grid:
        return patches  # [B, H//h, W//w, C, h, w]

    # Flatten spatial grid
    B, Hh, Ww, C, ph, pw = patches.shape
    patches = patches.reshape(B, Hh * Ww, C, ph, pw)  # [B, num_patches, C, h, w]

    return patches


def assemble_patches(patches, h, w):
    # Assert input is a 4D tensor [N, C, ph, pw]
    assert isinstance(patches, torch.Tensor), "Input must be a torch.Tensor"
    assert patches.ndim in (
        4,
        5,
    ), f"Expected 4D tensor [N, C, ph, pw], got shape {patches.shape}"

    if patches.ndim == 5:
        patches = patches.reshape(-1, *patches.shape[2:])

    N, C, ph, pw = patches.shape

    # Assert patch grid size is valid
    assert h > 0 and w > 0, f"Grid shape must be positive, got h={h}, w={w}"
    assert h * w == N, f"Expected {h * w} patches for a {h}x{w} grid, but got {N}"

    # Reshape to grid layout and assemble
    patches = patches.reshape(h, w, C, ph, pw)  # [h, w, C, ph, pw]
    patches = patches.permute(2, 0, 3, 1, 4)  # [C, h, ph, w, pw]
    img = patches.reshape(C, h * ph, w * pw)  # [C, H, W]
    return img
