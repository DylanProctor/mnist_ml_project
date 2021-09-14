import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import h5py
from tensorflow.keras.preprocessing.image import img_to_array
import time

model_path = 'tflite_final_model.tflite'
interpreter = tflite.Interpreter(model_path)
interpreter.allocate_tensors()

canvas = np.ones((268,268), dtype = "uint8") * 255
canvas[50:218,50:218] = 0

start_point = None
end_point = None
is_drawing = False

def draw_line(img, start_at, end_at):
    cv2.line(img, start_at, end_at, 255, 5)

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
    img = cv2.resize(img, (28,28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img = img/255.0
    return img    

def classify_digit(interpreter, img):
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
    interpreter.invoke()
    result = interpreter.tensor(interpreter.get_output_details()[0]["index"])()[0]
    digit = np.argmax(result)
    return digit

print()
print("Pay no attention to the error message above, I assure you I work perfectly.")
print("Hello, I am the amazing digit classifier robot")
print()
print("I can properly identify any digit between 0-9 you draw on that black square")
print("Pressing the 's' key on your keyboard allows you to draw using your mouse and the 'c' clears the baord")
print("The controls are little odd but just note that when you first press 's' it will draw a line coming from the top left corner")
print("Have no fear, just simply start where you want to start drawing the number and then press 'c' key to clear the baord and draw out the desired digit")
print("When you are finished press the 'p' to see my amazing prediction of the number")
print()
print("I trained myself off of 70,000 images of hand drawn digits so I am pretty experienced at this.")
print("That being said, I was only trained to identify numbers that are right side up, however if presented I will do my best to identify it ")
print()
print("When you have had enough fun press the 'q' key to exit the program")
print()


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
        canvas[50:218, 50:218] = 0
    elif key == ord('p'):
        image = canvas[50:218, 50:218]
        image = clean_img(image)
        digit = classify_digit(interpreter, image)
        print()
        print("Mmm...")
        time.sleep(1)
        print("I think it is a ", digit)
        print()

print()
print("Thank you for trying out the amazing digit classifier robot")
print()

cv2.destoryAllWindows()
