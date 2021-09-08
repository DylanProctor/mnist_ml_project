import cv2
import numpy as np
from tensorflow.keras.models import load_model
#import h5py
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('final_model.h5')

canvas = np.ones((48, 48), dtype = "uint8") * 255
canvas[10:38, 10:38] = 0

start_point = None
end_point = None
is_drawing = False

def draw_line(img, start_at, end_at):
    cv2.line(img, start_at, end_at, 255, 1)

def on_mouse_events(event, x, y, flags, params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_drawing:
            start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x,y)
            draw_line(canvas, start_point, end_point, )
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

def clean_img(img):
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img/255.0
    return img


cv2.namedWindow("Test Canvas")
cv2.setMouseCallback("Test Canvas", on_mouse_events)

while(True):
    cv2.imshow("Test Canvas", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_drawing = True
    elif key == ord('c'):
        canvas[10:38, 10:38] = 0
    elif key == ord('p'):
        image = canvas[10:38, 10:38]
        image = clean_img(image)
        result = model.predict_classes(image)
        print()
        print("Prediction: ", result[0])
        print()

cv2.destoryAllWindows()