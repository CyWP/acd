import torch
import torch.nn as nn
import numpy as np

PI2 = 2.0 * np.pi

class HueRotate(nn.Module):

    def __init__(self, img_size=(64, 64), channels=[8, 16, 8, 4], 
                 rwgt=0.3086, gwgt=0.6094, bwgt=0.114):
        super().__init__()
        self.rwgt = rwgt
        self.gwgt = gwgt
        self.bwgt = bwgt

        # === Compute transformation matrices ===
        sq2 = 2.0 ** 0.5
        sq3 = 3.0 ** 0.5

        # RGB to Z-aligned coordinate space
        g2zx = torch.tensor([
            [1, 0, 0],
            [0, 1/sq2, -1/sq2],
            [0, 1/sq2,  1/sq2]
        ])
        g2zy = torch.tensor([
            [sq2/sq3, 0, -1/sq3],
            [0,       1, 0],
            [1/sq3,   0, sq2/sq3]
        ])
        g2z = g2zx @ g2zy

        lum = g2z @ torch.tensor([rwgt, gwgt, bwgt])

        zsx = lum[0] / lum[2]
        zsy = lum[1] / lum[2]

        shear = torch.tensor([
            [1, 0, -zsx],
            [0, 1, -zsy],
            [0, 0, 1]
        ])
        unshear = torch.tensor([
            [1, 0, zsx],
            [0, 1, zsy],
            [0, 0, 1]
        ])

        # Z-aligned to RGB space
        zx2g = torch.tensor([
            [1, 0, 0],
            [0, 1/sq2,  1/sq2],
            [0, -1/sq2, 1/sq2]
        ])
        zy2g = torch.tensor([
            [sq2/sq3, 0,  1/sq3],
            [0,       1,  0],
            [-1/sq3,  0, sq2/sq3]
        ])
        z2g = zy2g @ zx2g

        # Store only the composed transforms
        self.register_buffer("pre_rot", g2z @ shear)
        self.register_buffer("post_rot", unshear @ z2g)

        # === CNN for generating rotation angle map ===
        conv_channels = [3] + channels + [1]
        layers = []
        for in_c, out_c in zip(conv_channels[:-1], conv_channels[1:]):
            layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, padding=1))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        rotations = torch.sin(self.convs(x))  # values in [-1, 1]
        return self.rotate_hues(rotations, x)

    def rotate_hues(self, rotations, img):
        """
        Apply per-pixel hue rotation using 3x3 linear transforms.
        """
        theta = rotations * PI2  # scale to [−π, π]
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # Build per-pixel Z-rotation matrices
        R = torch.zeros((*theta.shape, 3, 3), device=theta.device)
        R[..., 0, 0] = cos_t[:, 0]
        R[..., 0, 1] = -sin_t[:, 0]
        R[..., 1, 0] = sin_t[:, 0]
        R[..., 1, 1] = cos_t[:, 0]
        R[..., 2, 2] = 1.0

        # Compose full 3x3 transformation per pixel
        # T = post_rot @ R @ pre_rot
        pre = self.pre_rot.to(img.device)
        post = self.post_rot.to(img.device)
        # T = torch.einsum('ij,bhwjk->bhwik', post, torch.einsum('bhwij,jk->bhwik', R, pre))
        T = torch.matmul(torch.matmul(pre, R), post).squeeze(0)
        #breakpoint()
        # Apply transform to each pixel's RGB vector
        img = img.permute(0, 2, 3, 1).unsqueeze(-1)      # [B, H, W, 3, 1]
        out = torch.matmul(T, img).squeeze(-1)           # [B, H, W, 3]
        return out.permute(0, 3, 1, 2)                   # [B, 3, H, W]


if __name__ == "__main__":
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import matplotlib.pyplot as plt
    import sys

    for angle in [0, 0.25, -0.25, 0.5, -0.5]:
        # --- Load image ---
        img_path = "/home/cyvvp/Pictures/mip_bg.png"
        image = Image.open(img_path).convert("RGB")

        transform = T.Compose([
            T.Resize((64, 64)),  # Resize for speed
            T.ToTensor(),          # [C, H, W] in [0,1]
        ])
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension -> [1, 3, H, W]

        # --- Dummy rotation field: rotate everything by 90 degrees (π/2) ---
        B, C, H, W = img_tensor.shape
        rotations = torch.full((1, 1, H, W), angle)

        # --- Apply hue rotation ---
        model = HueRotate()
        with torch.no_grad():
            out = model.rotate_hues(rotations, img_tensor)

        # --- Show result ---
        to_pil = T.ToPILImage()
        result_img = to_pil(out.squeeze(0).clamp(0, 1))
        result_img.show()