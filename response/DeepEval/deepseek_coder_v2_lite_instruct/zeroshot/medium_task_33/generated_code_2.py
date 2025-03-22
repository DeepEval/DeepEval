import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Lambda, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three channel groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction for each channel group
    conv1x1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    conv3x3 = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv5x5 = Conv2D(32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])

    # Concatenate the outputs from the three groups
    concat_layer = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5])

    # Flatten the concatenated output
    flatten_layer = Flatten()(concat_layer)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()