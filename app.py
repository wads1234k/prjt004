from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("model\\keras_model.h5", compile=False)
# Load the labels
class_names = open("model\\labels.txt", "r").readlines()



camera = cv2.VideoCapture(0)  # 0은 기본 웹캠

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # 이미지가 제대로 읽어졌는지 확인
    if not ret:
        print("Failed to grab image")
        break

    # Resize the raw image into (224-height, 224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the models input shape.
    image_input = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_input = (image_input / 127.5) - 1

    # Predict the model
    prediction = model.predict(image_input)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score in the console
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Display prediction and confidence score on the image
    text = f"Class: {class_name[2:-1]}, Conf: {np.round(confidence_score * 100)}%"
    
    # Make sure the text is readable (yellow text)
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Show the image in a window (This is done only once now)
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key on your keyboard
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()