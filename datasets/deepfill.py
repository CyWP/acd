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


class DeepFill(AcDataset):

    _processed_folder_name = "processed"

    def __init__(
        self,
        folder_path,
        patch_size=64,
        region_size=196,
        seed=42,
        preprocess=True,
        blank_ctl_p=0.15,
        blank_nbhd_p=0.25,
        copy_to=None,
        scale=1.0,
        num_reps=100,
    ):
        """
        folder_path: Folder of preprocessed image patch tensors and control data.
        patch_size: Tuple of (H, W) per patch.
        transform: torchvision transforms applied to both patch and neighborhood.
        seed: Optional seed for reproducibility.
        """
        self.folder = folder_path
        self.patch_size = patch_size
        self.seed = seed
        self.num_reps = num_reps
        process_name = f"{PatchNeighborhoodDataset._processed_folder_name}_s{scale}"
        if seed is not None:
            random.seed(seed)
            torch.seed(seed)
        if preprocess:
            print("preprocessing")
            preprocess_folder(
                self.folder, process_name, get_hed(self.folder), patch_size, scale=scale
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
        self.blank_ctl_p = blank_ctl_p
        self.blank_nbhd_p = blank_nbhd_p

    def __len__(self):
        return len(self.patch_indices)

    def get_sample(self, idx):
        path = self.file_paths[idx%len(self.file_paths)]
        tensor_data = torch.load(path)  # load both patches and control data
        img, ctl = tensor_data["img"].to(get_device()), tensor_data["ctl"].to(get_device())
        h, w = img.shape[1], img.shape[2]
        tc, lc = random.randint(0, h-self.region_size), random.randint(0, w-self.region_size)
        img_patch = img[:, tc:tc+self.region_size, lc:lc+self.region_size]
        ctl_patch = ctl[:, tc:tc+self.region_size, lc:lc+self.region_size]
        return AcDataSample(
            {
                IMG_KEY: (center_patch * 2 - 1).to(get_device()),
                NBHD_KEY: (nbhd_patch * 2 - 1).to(get_device()),
                CTL_KEY: (center_ctl * 2 - 1).to(get_device()),
            }
        )


def preprocess_folder(
    folder_path, tensor_path, ctl_encoder, patch_size=(64, 64), scale=1.0
):
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
        img_data = transform(image)  # shape: [C, H, W]

        # Extract ControlNet features (e.g., HED maps)
        control_data = ctl_encoder(img_data)

        # Save both the patches and control data
        out_path = os.path.join(save_dir, fname.rsplit(".", 1)[0])
        torch.save({"img": img_data, "ctl": control_data}, out_path + ".pt")

        print(
            f"Saved patches and control data for '{fname}' to '{out_path}.pt'. Shapes: {patches.shape}, {control_data.shape}"
        )
