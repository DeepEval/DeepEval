import tensorflow as tf
from tensorflow.keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, SeparableConv2D, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block with different scale max pooling
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flat1 = Flatten()(pool1)

    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flat2 = Flatten()(pool2)

    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flat3 = Flatten()(pool3)

    # Concatenate flattened outputs and apply dropout
    concat1 = Concatenate()([flat1, flat2, flat3])
    dropout1 = Dropout(0.5)(concat1)

    # Fully connected layer and reshape operation
    dense1 = Dense(1024, activation='relu')(dropout1)
    reshape1 = Reshape((8, 8, 16))(dense1)  # Reshape to a 4D tensor

    # Second block with tf.split and separable convolutions
    def split_and_conv(x):
        # Split into 4 groups
        split = tf.split(x, num_or_size_splits=4, axis=-1)

        # Process each group with different kernel sizes
        conv1 = SeparableConv2D(16, (1, 1), activation='relu', padding='same')(split[0])
        conv2 = SeparableConv2D(16, (3, 3), activation='relu', padding='same')(split[1])
        conv3 = SeparableConv2D(16, (5, 5), activation='relu', padding='same')(split[2])
        conv4 = SeparableConv2D(16, (7, 7), activation='relu', padding='same')(split[3])

        # Concatenate outputs
        concat2 = Concatenate()([conv1, conv2, conv3, conv4])
        return concat2

    processed_groups = Lambda(split_and_conv)(reshape1)

    # Flatten and final fully connected layer for classification
    flat_final = Flatten()(processed_groups)
    output = Dense(10, activation='softmax')(flat_final)

    # Create the model
    model = Model(inputs=input_layer, outputs=output)

    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()