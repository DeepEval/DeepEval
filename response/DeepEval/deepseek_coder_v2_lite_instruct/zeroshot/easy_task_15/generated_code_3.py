import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input layer
    inputs = Input(shape=(28, 28, 1))

    # Block 1
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(32, kernel_size=(1, 1), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(64, kernel_size=(1, 1), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Flatten layer
    x = Flatten()(x)

    # Fully connected layer
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model