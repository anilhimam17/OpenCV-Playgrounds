import cv2

# Initialising the Capture Object
capture = cv2.VideoCapture(0)


# Function to darw a line on the screen
def draw_a_line() -> None:
    while True:
        # Reading the frames from the livestream
        ret, frame = capture.read()

        # Retreiving the properties of the frames from the livestream
        width = int(capture.get(3))
        height = int(capture.get(4))

        # Update Image with the Line
        img_updated = cv2.line(frame, (0, 0), (width, height), (255, 0, 0), 3)
        img_updated = cv2.line(frame, (0, height), (width, 0), (255, 255, 0), 3)

        # Displaying the image
        cv2.imshow("Image with Line", img_updated)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


# Function to draw a circle
def draw_a_circle(radius: int) -> None:
    while True:
        # Reading the frames from the livestream
        ret, frame = capture.read()

        # Retrieving the properties from the livestream
        width = int(capture.get(3))
        height = int(capture.get(4))

        # Updating the Image
        img_updated = cv2.circle(frame, (width // 2, height // 2), radius, (255, 0, 255), 3)

        # Displaying the Image
        cv2.imshow("Image with Circle", img_updated)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


# Function to draw a rectangle
def draw_a_rectangle() -> None:
    while True:
        # Reading the Frames
        ret, frame = capture.read()

        # Retrieving the Properties
        width = int(capture.get(3))
        height = int(capture.get(4))

        # Updating the Image
        img_updated = cv2.rectangle(
            img=frame, pt1=(int(width * (1/4)), int(height * (1/4))),
            pt2=(int(width * (3/4)), int(height * (3/4))), color=(0, 255, 255),
            thickness=10
        )

        # Display the image
        cv2.imshow("Image with Rectangle", img_updated)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


# Function to draw custom text onto the screen
def draw_text(text: str) -> None:
    while True:
        # Returning the Frames
        ret, frame = capture.read()

        # Returning the Properties
        width = int(capture.get(3))
        height = int(capture.get(4))

        # Setting the Font Style
        font_style = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

        # Updating the image
        img_updated = cv2.putText(
            img=frame, text=text, org=(width - 900, height - 500),
            fontFace=font_style, fontScale=3, color=(255, 0, 255), thickness=2,
            lineType=cv2.LINE_AA
        )

        # Displaying the Image
        cv2.imshow("Image with text", img_updated)
        if cv2.waitKey(1) == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


draw_text("Hello, OpenCV")
