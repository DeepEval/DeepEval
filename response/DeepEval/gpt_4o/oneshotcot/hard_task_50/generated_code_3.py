import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, Concatenate, SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Multiple MaxPooling layers with different scales
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    # Flatten each pooling output
    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    # Dropout to mitigate overfitting
    drop1 = Dropout(rate=0.5)(flat1)
    drop2 = Dropout(rate=0.5)(flat2)
    drop3 = Dropout(rate=0.5)(flat3)

    # Concatenate the dropout outputs
    concat = Concatenate()([drop1, drop2, drop3])

    # Fully connected layer
    dense = Dense(units=1024, activation='relu')(concat)

    # Reshape to prepare for the second block
    reshape = Reshape(target_shape=(16, 16, 4))(dense)

    # Second Block: Split and process each group
    def split_and_process(tensor):
        # Split into 4 groups
        splits = tf.split(tensor, num_or_size_splits=4, axis=-1)

        # Process each group with separable convolutions of varying kernel sizes
        conv1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        conv3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        conv4 = SeparableConv2D(filters=32, kernel_size=(7, 7), padding='same', activation='relu')(splits[3])

        # Concatenate the outputs
        return Concatenate()([conv1, conv2, conv3, conv4])

    # Encapsulate the split and process operation using a Lambda layer
    processed = Lambda(split_and_process)(reshape)

    # Flatten and final fully connected layer for classification
    flatten_layer = Flatten()(processed)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model