import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dense, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    def feature_extraction_block(input_tensor):
        # Split the input tensor into three groups (channels)
        split_layers = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)

        # Process each group through a separable convolutional layer
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])

        # Concatenate the outputs of the three groups
        concatenated = Concatenate()([conv1x1, conv3x3, conv5x5])

        return concatenated

    # Apply the feature extraction block
    extracted_features = feature_extraction_block(input_layer)

    # Flatten the concatenated output
    flattened = Flatten()(extracted_features)

    # Pass through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model