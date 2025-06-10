import cv2
import numpy as np

def reconstruct_image_from_gradient(grad_x, grad_y):
    # Initialize the reconstructed image
    h, w = grad_x.shape
    image = np.zeros((h, w))

    # Integrate along the x-axis (horizontal gradient)
    for y in range(h):
        image[y, 1:] = np.cumsum(grad_x[y, :-1])

    # Integrate along the y-axis (vertical gradient)
    for x in range(1, w):
        image[1:, x] += np.cumsum(grad_y[:-1, x])

    # Normalize the image (optional)
    image -= image.min()
    image /= image.max()
    return image

if __name__ == "__main__":

    # # Load the image and create a mask for the missing region
    # image = cv2.imread("/home/cyvvp/Pictures/manfishblock.jpg")
    # mask = cv2.imread(
    #     "/home/cyvvp/Pictures/manfishmask_inv.jpg", cv2.IMREAD_GRAYSCALE
    # )  # Mask: 255 for missing regions, 0 for known regions

    # # Navier-Stokes Inpainting
    # navier_stokes_result = cv2.inpaint(
    #     image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS
    # )

    # # Telea Inpainting
    # telea_result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # # Display the results
    # cv2.imshow("Original", image)
    # cv2.imshow("Navier-Stokes Inpainting", navier_stokes_result)
    # cv2.imshow("Telea Inpainting", telea_result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
