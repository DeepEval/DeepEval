# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import applications
from tensorflow.keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Part 1: Feature extraction
    input_layer = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)

    # Part 2: Generalization enhancement
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    dropout1 = layers.Dropout(0.2)(conv4)
    conv5 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(dropout1)

    # Part 3: Upsampling with skip connections
    conv6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    transpose_conv1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu')(conv6)
    concat1 = layers.Concatenate()([transpose_conv1, conv3])
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat1)

    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    transpose_conv2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu')(conv8)
    concat2 = layers.Concatenate()([transpose_conv2, conv2])
    conv9 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat2)

    conv10 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv9)
    transpose_conv3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu')(conv10)
    concat3 = layers.Concatenate()([transpose_conv3, conv1])
    conv11 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat3)

    # Output layer
    output_layer = layers.Conv2D(10, (1, 1), activation='softmax')(conv11)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model