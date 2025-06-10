import cv2
import numpy as np

from numpy.fft import fft2, ifft2, fftshift, ifftshift


def soft_threshold(x, lam):
    """Soft thresholding operator."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


import numpy as np
from numpy.fft import fft2, ifft2


def fourier_inpaint_rgb_pocs(image, mask, n_iter=200):
    """
    Basic POCS-style inpainting using FFT.
    Parameters:
        image : RGB float32 array (H, W, 3)
        mask : 2D binary array (H, W), 1 = known, 0 = missing
        n_iter : Number of iterations

    Returns:
        Inpainted image (H, W, 3)
    """
    H, W, C = image.shape
    inpainted = image.copy()

    for c in range(3):
        x = inpainted[..., c]
        x[mask == 0] = 0  # initialize missing pixels

        for i in range(n_iter):
            X = fft2(x)
            x_rec = np.real(ifft2(X))
            x_rec[mask == 1] = image[..., c][mask == 1]  # replace known pixels
            x = x_rec

        inpainted[..., c] = np.clip(x, 0, 1)

    return inpainted


if __name__ == "__main__":
    # Load the color image and mask
    image = cv2.imread("/home/cyvvp/Pictures/manfishblock.jpg")
    image = image / 255
    mask = cv2.imread("/home/cyvvp/Pictures/manfishmask.jpg", cv2.IMREAD_GRAYSCALE)
    mask = mask > 0

    reconstructed_image = fourier_inpaint_rgb_pocs(image, mask)
    # Display the results
    cv2.imshow("Original", image)
    cv2.imshow("Reconstructed Image", reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
