import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Create four parallel convolutional paths
    path1 = Conv2D(16, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(16, (1, 1), activation='relu')(inputs)
    path2 = Conv2D(32, (3, 3), activation='relu')(path2)
    path3 = Conv2D(16, (1, 1), activation='relu')(inputs)
    path3 = Conv2D(32, (3, 3), activation='relu')(path3)
    path4 = MaxPooling2D((2, 2))(inputs)
    path4 = Conv2D(16, (1, 1), activation='relu')(path4)

    # Concatenate the outputs from the parallel paths
    concat = concatenate([path1, path2, path3, path4])

    # Flatten and pass through a dense layer
    flatten = tf.keras.layers.Flatten()(concat)
    dense = Dense(128, activation='relu')(flatten)

    # Output layer
    outputs = Dense(10, activation='softmax')(dense)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model