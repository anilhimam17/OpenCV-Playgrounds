# Importing OpenCV
from typing import Any
import cv2
from cv2.typing import MatLike

# Loading an image
IMG_PATH = "./assets/shinobu-kocho-demon-slayer-kimetsu-no-yaiba-3840x2160-7913.jpg"


# Function to display an image
def display_image(path: str) -> None:
    """
    Loading an Image:
    Takes an option for a custom mode to load the image.
    Since by default it loads the image in Blue-Gree-Red channel format over Red-Green-Blue.

    Modes:
    - cv2.IMREAD_COLOR: -1 (Alpha Channel is neglected)
    - cv2.IMREAD_GRAYSCALE: 0 (Loads the image in grayscale)
    - cv2.IMREAD_UNCHANGED: 1 (Alpha Channel is included)
    """
    img = cv2.imread(path, 0)

    # Displaying the Image
    cv2.imshow("Image Display", img)

    # Auto destroys the display image after a set delay incase no key is pressed.
    _ = cv2.waitKey(5000)
    cv2.destroyAllWindows()


# Function to resize the image and display
def resize_img(path: str, size: Any) -> None:
    """
    Resize can shape the image based on:
    - Specific shape passed into the resize().
    - A fraction of the image based on the init coordinates of the image. 
    """

    img = cv2.imread(path)
    img_resize = cv2.resize(img, (0, 0), fx=size[0], fy=size[-1])

    cv2.imshow("Resized Image", img_resize)
    _ = cv2.waitKey(5000)
    cv2.destroyAllWindows()


def rotate_img(path: str) -> None:
    """
    Rotation of a Image can utilise one of two ways:
    - Utilisation of hardcoded predefined angles from cv2.cv2
    - Utilisation of the rotateCode param from cv2.rotate
    """

    img = cv2.imread(path, 1)
    img_rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    cv2.imshow("Rotated Image", img_rotate)
    _ = cv2.waitKey(5000)
    cv2.destroyAllWindows()


def save_and_load_img(img: MatLike) -> None:
    
    # Writing any changes made to the image into a seperate file
    _ = cv2.imwrite("./assets/transformed_img.jpg", img)
    loaded_img = cv2.imread("./assets/transformed_img.jpg")
    cv2.imshow("Transformed Image", loaded_img)
    _ = cv2.waitKey(5000)
    cv2.destroyAllWindows()


img = cv2.imread(IMG_PATH, 0)
save_and_load_img(img)
