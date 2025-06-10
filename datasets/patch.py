import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import shutil
import cv2

from PIL import Image, ImageFilter

from ..utils import IMG_KEY, CTL_KEY, NBHD_KEY
from ..utils.pytorch import AcDataSample, AcDataset, get_device
from ..networks.hed import get_hed


class PatchNeighborhoodDataset(AcDataset):

    _processed_folder_name = "processed"

    def __init__(
        self,
        folder_path,
        patch_size=(64, 64),
        seed=42,
        preprocess=True,
        blank_ctl_p=0.15,
        blank_nbhd_p=0.25,
        copy_to=None,
        scale=1.0,
    ):
        """
        folder_path: Folder of preprocessed image patch tensors and control data.
        patch_size: Tuple of (H, W) per patch.
        transform: torchvision transforms applied to both patch and neighborhood.
        seed: Optional seed for reproducibility.
        """
        self.folder = folder_path
        self.patch_size = patch_size
        self.form_transform = PairedFormalAugmentation()
        self.color_transform = ColorAugmentation()
        self.seed = seed
        self.ctl_encoder = get_hed(self.folder)
        process_name = f"{PatchNeighborhoodDataset._processed_folder_name}_s{scale}"
        if seed is not None:
            random.seed(seed)
        if preprocess:
            print("preprocessing")
            preprocess_folder(
                self.folder, process_name, self.ctl_encoder, patch_size, scale=scale
            )
            self.folder = os.path.join(self.folder, process_name)
        if copy_to:
            copy_folder = os.path.join(copy_to, process_name)
            print(f"Copying to: {copy_to}")
            shutil.copytree(self.folder, copy_folder, dirs_exist_ok=True)
            self.folder = copy_folder
        self.file_paths = [
            os.path.join(self.folder, f)
            for f in os.listdir(self.folder)
            if f.endswith(".pt")
        ]
        self.patch_indices = self._get_patch_indices()
        self.blank_ctl_p = blank_ctl_p
        self.blank_nbhd_p = blank_nbhd_p

    def _get_patch_indices(self):
        print("Indexing dataset")
        indices = []
        for path in self.file_paths:
            tensor_data = torch.load(path)  # load both patches and control data
            patches = tensor_data["patches"]
            rows, cols = patches.shape[:2]  # Get the grid size (H_grid, W_grid)

            # Ensure the center patch is not within 2 patches from the edge
            for i in range(2, rows - 2):  # Start from index 2 and end at rows - 2
                for j in range(2, cols - 2):  # Start from index 2 and end at cols - 2
                    indices.append((path, i, j))
        return indices

    def __len__(self):
        return len(self.patch_indices)

    def get_sample(self, idx):
        path, i, j = self.patch_indices[idx]
        tensor_data = torch.load(path)  # load both patches and control data
        patches = tensor_data["patches"].to(get_device())
        control_data = tensor_data["control_data"].to(get_device())

        patch_h, patch_w = self.patch_size

        # 1. Construct the 2-ring neighborhood (5x5 grid, including center)
        full_grid = torch.cat(
            [
                torch.cat([patches[i + di, j + dj] for dj in range(-2, 3)], dim=-1)
                for di in range(-2, 3)
            ],
            dim=-2,
        )

        full_control_data = torch.cat(
            [
                torch.cat([control_data[i + di, j + dj] for dj in range(-2, 3)], dim=-1)
                for di in range(-2, 3)
            ],
            dim=-2,
        )

        if self.form_transform:
            torch.manual_seed(self.seed)
            full_grid, full_control_data = self.form_transform(
                full_grid, full_control_data
            )
        if self.color_transform:
            torch.manual_seed(self.seed)
            full_grid = self.color_transform(full_grid)

        # Apply random translation of the centroid (within ±patch_size/2)
        max_translation = patch_h // 2
        tx = random.randint(-max_translation, max_translation)
        ty = random.randint(-max_translation, max_translation)

        # Extract center patches
        center_patch = full_grid[
            :, 2 * patch_h + ty : 3 * patch_h + ty, 2 * patch_w + tx : 3 * patch_w + tx
        ].clone()

        # Randomly blank out ctl data
        if random.random() > self.blank_ctl_p:
            center_ctl = full_control_data[
                :,
                2 * patch_h + ty : 3 * patch_h + ty,
                2 * patch_w + tx : 3 * patch_w + tx,
            ]
        else:
            center_ctl = torch.zeros((1, *center_patch.shape[1:]))

        if random.random() > self.blank_nbhd_p:
            # Extract neighborhood patch
            nbhd_patch = full_grid[
                :, patch_h + ty : 4 * patch_h + ty, patch_w + tx : 4 * patch_w + tx
            ]

            # Randomly blank out patches in the 3x3 neighborhood
            k = random.randint(1, 8)
            positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
            blanked = random.sample(positions, k)
            blanked.append((1, 1))  # Always blank out center
            for row, col in blanked:
                ys = row * patch_h
                xs = col * patch_w
                nbhd_patch[:, ys : ys + patch_h, xs : xs + patch_w] = 0
        else:
            nbhd_patch = torch.zeros((full_grid.shape[0], 3 * patch_h, 3 * patch_w))

        return AcDataSample(
            {
                IMG_KEY: (center_patch * 2 - 1).to(get_device()),
                NBHD_KEY: (nbhd_patch * 2 - 1).to(get_device()),
                CTL_KEY: (center_ctl * 2 - 1).to(get_device()),
            }
        )


class PairedFormalAugmentation:
    def __init__(
        self,
        patch_size=(320, 320),
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotation_deg=20,
        scale_range=(0.8, 1.2),
        shear_deg=15,
        translate_frac=0.1,
    ):
        self.patch_size = patch_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotation_deg = rotation_deg
        self.scale_range = scale_range
        self.shear_deg = shear_deg
        self.translate_frac = translate_frac

    def __call__(self, img1, img2):
        # Flip
        do_hflip = random.random() < self.hflip_prob
        do_vflip = random.random() < self.vflip_prob

        if do_hflip:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)
        if do_vflip:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        # Rotation
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle)

        # Affine (angle=0 since rotation is separate)
        translate = [
            random.uniform(-self.translate_frac, self.translate_frac)
            * self.patch_size[0]
            for _ in range(2)
        ]
        scale = random.uniform(*self.scale_range)
        shear = [random.uniform(-self.shear_deg, self.shear_deg) for _ in range(2)]

        img1 = TF.affine(img1, angle=0, translate=translate, scale=scale, shear=shear)
        img2 = TF.affine(img2, angle=0, translate=translate, scale=scale, shear=shear)

        return img1, img2


class ColorAugmentation:
    def __init__(self, brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05):
        self.transform = transforms.Compose(
            [
                # Random color jitter (moderate color variations)
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
            ]
        )

    def __call__(self, img):
        return img
        return self.transform(img)


def preprocess_folder(
    folder_path, tensor_path, ctl_encoder, patch_size=(64, 64), scale=1.0
):
    """
    Processes each image in `folder_path` and saves a 2D grid of patches in [-1, 1]
    to `folder_path/tensor_path/<image>.pt`, and also extracts and saves ControlNet features.

    Output tensor shape: (H_grid, W_grid, C, patch_h, patch_w)
    """
    save_dir = os.path.join(folder_path, tensor_path)
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.ToTensor()  # Converts PIL image to [0, 1] tensor in CHW

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff", "webp")
        ):
            continue

        img_path = os.path.join(folder_path, fname)
        image = Image.open(img_path).convert("RGB")
        new_size = (int(image.width * scale), int(image.height * scale))
        if scale < 1.0:
            # Rough rule: radius ~ (1/scale - 1), capped at a reasonable max
            radius = min(5.0, max(0.0, (1.0 / scale - 1)))
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        image = image.resize(new_size, Image.LANCZOS)
        tensor = transform(image)  # shape: [C, H, W]
        tensor = tensor.to(get_device())

        c, h, w = tensor.shape
        ph, pw = patch_size

        # Pad so h % ph == 0 and w % pw == 0
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

        # Extract 2D grid of patches
        patches = tensor.unfold(1, ph, ph).unfold(
            2, pw, pw
        )  # [C, H//ph, W//pw, ph, pw]
        patches = patches.permute(1, 2, 0, 3, 4)  # [H_grid, W_grid, C, ph, pw]

        # Flatten the first two dimensions (H_grid, W_grid)
        patches_flattened = patches.reshape(
            -1, patches.shape[2], patches.shape[3], patches.shape[4]
        )

        # Extract ControlNet features (e.g., HED maps)
        control_data = ctl_encoder(patches_flattened)

        # Reshape back to the original form (H_grid, W_grid, C, ph, pw)
        if control_data is not None:
            control_data = control_data.reshape(
                patches.shape[0],
                patches.shape[1],
                1,
                patches.shape[3],
                patches.shape[4],
            )

        # Save both the patches and control data
        out_path = os.path.join(save_dir, fname.rsplit(".", 1)[0])
        torch.save({"patches": patches, "control_data": control_data}, out_path + ".pt")

        print(
            f"Saved patches and control data for '{fname}' to '{out_path}.pt'. Shapes: {patches.shape}, {control_data.shape}"
        )
