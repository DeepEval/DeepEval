import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def method():

    # Load and preprocess the image data
    image = cv2.imread('image.jpg')
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0

    # Create the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(np.expand_dims(image, axis=0), np.zeros((1, 10)), epochs=1)

    # Predict the image
    prediction = model.predict(np.expand_dims(image, axis=0))

    # Return the prediction
    return prediction

# Call the method
output = method()

# Print the output
print(output)