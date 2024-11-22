import numpy as np
import cv2
from mss import mss
from pynput.mouse import Listener

cont = 0
pos = {"x": [], "y": []}

def is_clicked(x, y, button, pressed):
    global cont, pos
    if pressed:
        print('Clicked!')
        pos["x"].append(x)
        pos["y"].append(y)
        cont += 1
        if cont == 2:
            return False  # Stop listener after two clicks

# Start the listener
with Listener(on_click=is_clicked) as listener:
    listener.join()

# Ensure we have two points for the bounding box
if len(pos["x"]) == 2 and len(pos["y"]) == 2:
    # Create bounding box based on clicks
    bounding_box = {
        'top': min(pos["y"]),
        'left': min(pos["x"]),
        'width': abs(pos["x"][1] - pos["x"][0]),
        'height': abs(pos["y"][1] - pos["y"][0])
    }
else:
    print("Error: Two clicks are required to define the bounding box.")
    exit(1)

# Initialize mss
sct = mss()

while True:
    try:
        # Capture the screen
        sct_img = sct.grab(bounding_box)
        screen_np = np.array(sct_img)

        # Display the screen capture
        cv2.imshow('Screen Capture', screen_np)

        # Exit when 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break
    except Exception as e:
        print(f"An error occurred: {e}")
        break