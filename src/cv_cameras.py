import cv2
import numpy as np

# Initialising a Video Capture Object in OpenCV
"""
The video capture object can be overloaded for:
- Usage on multiple webcams.
- Usage on a livestream.
- Usage on a video.

0: By default leads to the first accessible video stream on device
"""
capture = cv2.VideoCapture(0)


def livestream() -> None:
    while True:
        """
        Returns:
        - A status of capture stream: ret.
        - A frame retrieved from the capture stream: frame.
        """

        ret, frame = capture.read()

        # Displaying the Frame
        cv2.imshow("Video Stream", frame)

        # Break the loop if q is pressed within 1ms of each frame, for better fluidity in frames
        if cv2.waitKey(1) == ord("q"):
            break

    # Releasing the Camera Resource
    capture.release()
    cv2.destroyAllWindows()


def livestream_quad() -> None:
    while True:
        # Capturing feed from the livestream
        ret, frame = capture.read()

        # Acquing width and height of the image from the properties
        width = int(capture.get(3))
        height = int(capture.get(4))

        # Creating a new frame to partition and relay the frames as a quad
        new_image = np.zeros_like(frame, dtype=np.uint8)

        # Creating a smaller version of the frame
        smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Overlaying the smaller frames into the custom image feed
        new_image[:height//2, :width//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)  # Quad - 2
        new_image[:height//2, width//2:] = smaller_frame  # Quad - 1
        new_image[height//2:, width//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)  # Quad - 4
        new_image[height//2:, :width//2] = smaller_frame  # Quad - 3

        # Showing the Image
        cv2.imshow("Quad Livestream", new_image)

        # Breaking condition
        if cv2.waitKey(1) == ord("q"):
            break

    # Releasing the Camera Resource
    capture.release()
    cv2.destroyAllWindows()


livestream_quad()
