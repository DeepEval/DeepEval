import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    split_1, split_2, split_3 = tf.split(input_layer, num_or_size_splits=3, axis=-1)

    # Feature extraction through separable convolutional layers
    conv_1x1 = Conv2D(32, (1, 1), activation='relu')(split_1)
    conv_3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_2)
    conv_5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_3)

    # Concatenate the outputs from the three groups
    concatenated = Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense_1 = Dense(128, activation='relu')(flattened)
    dense_2 = Dense(64, activation='relu')(dense_1)
    output_layer = Dense(10, activation='softmax')(dense_2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()