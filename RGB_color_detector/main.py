import cv2
import numpy as np

# Takeing the input from webcam
vid = cv2.VideoCapture(0)

# Running while loop just to make sure that our prog keeps running 
# until we stop it

while True:
    # Capturing the current frame
    _, frame = vid.read()

    # Displaying the current frame
    cv2.imshow("frame", frame)

    # Setting values for base colors
    b = frame[:, :, 0] # Blue channel
    g = frame[:, :, 1] # Green channel
    r = frame[:, :, 2] # Red channel

    # Computing the mean
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    # Display the most prominent color
    if b_mean > g_mean and b_mean > r_mean:
        print("The displayed color is Blue")
    elif g_mean > b_mean and g_mean > r_mean:
        print("The displayed color is Green")
    else:
        print("The displayed color is Red")

    # Break the look if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the video capture object and closing all windows
vid.release()
cv2.destroyAllWindows()
