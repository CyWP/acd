import os

import torch

from .utils.data import tensors_to_pil, one_hot_to_idx, is_one_hot_all
from .utils.pytorch import AcDataSample, get_device
from .utils import IMG_KEY, LABEL_KEY


def acedm_sampler(
    net,
    x,
    gnet=None,
    num_steps=32,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    guidance=1,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
    dtype=torch.float32,
    randn_like=torch.randn_like,
):

    def pkg_input(x, additional_input):
        if additional_input:
            new = additional_input.copy()
            new.img = x
            return new
        return AcDataSample({IMG_KEY: x})

    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, return_logvar=False).to(dtype)
        if guidance == 1:
            return Dx
        if gnet:
            ref_Dx = gnet(x, t, return_logvar=False).to(dtype)  # Autoguidance
        else:
            x_copy = x.copy()
            x_copy.label = torch.zeros_like(x.label)
            ref_Dx = net(x_copy, t, return_logvar=False).to(dtype)  # CFG
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=x.noise.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    noise = x.noise

    x_next = noise * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur
        # Euler step.
        x.noise = x_hat
        d_cur = (x_hat - denoise(x, t_hat)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x.noise = x_next
            d_prime = (x_next - denoise(x, t_next)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def generate_images(
    net,
    x,
    num_images=1,
    gnet=None,  # Optional guidance network. Uses CFG if None.
    outdir=None,
    seed=42,
    save_input=True,
    guidance_strength=1,
    num_steps=32,
    noise=None,
    img_size=(64, 64),
):

    x.noise = torch.randn((num_images, 3, *img_size)).to(get_device())

    with torch.no_grad():
        denoised = acedm_sampler(
            net,
            x,
            gnet=gnet,
            guidance=guidance_strength,
            num_steps=num_steps,
        )

    if outdir:
        if save_input:
            for key, value in x.items():
                if key not in (IMG_KEY, LABEL_KEY):
                    for idx, img in enumerate(tensors_to_pil(value)):
                        img.save(os.path.join(outdir, f"{idx}{key}_input_img.png"))
        if is_one_hot_all(x.label):
            labels = one_hot_to_idx(x.label)
        else:
            labels = [""] * x.noise.shape[0]
        for idx, img in enumerate(tensors_to_pil(denoised)):
            img.save(
                os.path.join(outdir, f"{idx}img_label_{labels[idx]}_seed_{seed}.png")
            )

    return denoised
