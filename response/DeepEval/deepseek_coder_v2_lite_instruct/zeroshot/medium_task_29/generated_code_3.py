import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(conv1)

    # Second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)

    # Third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(conv3)

    # Flatten the output from each pooling layer
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)

    # Concatenate the flattened outputs
    combined_features = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layers
    dense1 = Dense(256, activation='relu')(combined_features)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=input_layer, outputs=dense2)

    return model