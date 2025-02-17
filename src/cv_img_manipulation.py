import cv2
import random

IMG_PATH = "./assets/wp11723676-initial-d-rx7-wallpapers.jpg"


# Loading Images
def load_image(image_path: str) -> None:
    """
    - Any Image loaded into OpenCV is immediately conformed to its numpy repr.
    - The numpy ndarray repr carries the values of each and every pixel.
    - The numpy ndarray also carries all the numpy array ops and properties.

    Each row and column in the numpy array repr is in the order:
    Blue-Green-Red to indicate the intensity of the colors in the pixels.
    """

    img = cv2.imread(image_path)
    print(img)


def add_random_noise_to_image(img_path: str) -> None:
    # Loading the Image
    img = cv2.imread(img_path)
    print(img.shape)

    # Adding random noise to the image by random pixel sampling
    for _ in range(1000000):
        row = random.randint(0, 1080 - 1)
        col = random.randint(0, 1920 - 1)
        img[row][col] = [
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ]

    cv2.imshow("Image with Random Noise", img)
    _ = cv2.waitKey(5000)
    cv2.destroyAllWindows()


add_random_noise_to_image(IMG_PATH)
