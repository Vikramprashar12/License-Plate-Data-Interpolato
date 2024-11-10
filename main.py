import cv2

from PIL import Image

from util import get_limits


def show_color_on_hover(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        color = frame[y, x]  # Get the BGR color at the cursor position
        print(f"Color at ({x}, {y}): BGR {color}")


yellow = [0, 255, 255]  # yellow in BGR colorspace
cap = cv2.VideoCapture(0)

# Set the desired resolution (e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


# Set the mouse callback function
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('frame', show_color_on_hover)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Resize the window to match the frame size and allow manual resizing
    cv2.resizeWindow('frame', frame.shape[1], frame.shape[0])

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
