import cv2
from keras.models import load_model
import numpy as np

# Load the trained models
mlp_model = load_model('mnist_mlp_model.h5')
cnn_model = load_model('mnist_cnn_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale if necessary
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Resize to 28x28 and invert colors
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = cv2.bitwise_not(resized)

    # Reshape for input to model
    reshaped = inverted.reshape(1, 28, 28, 1)

    return reshaped


# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the frame
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the rectangle on the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the digit from the original frame
        digit = gray[y:y+h, x:x+w]

        # Preprocess the digit image
        preprocessed_image = preprocess_image(digit)

        # Reshape the preprocessed image
        preprocessed_image = np.reshape(preprocessed_image, (1, 28, 28))

        # Make predictions using both models
        mlp_prediction = mlp_model.predict(preprocessed_image)
        cnn_prediction = cnn_model.predict(preprocessed_image)

        # Get the predicted labels for both models
        mlp_label = np.argmax(mlp_prediction)
        cnn_label = np.argmax(cnn_prediction)

        # Print the predicted labels
        print("MLP prediction: {}".format(mlp_label))
        print("CNN prediction: {}".format(cnn_label))

        # Draw the predicted label in the frame
        cv2.putText(frame, str(mlp_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Print the recognized number in the left-down corner of the frame
        cv2.putText(frame, f"Recognized Number: {mlp_label}", (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the original frame
    cv2.imshow("frame", frame)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
