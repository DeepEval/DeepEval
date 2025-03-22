import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Lambda, Concatenate, SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    drop1 = Dropout(rate=0.5)(flat1)
    drop2 = Dropout(rate=0.5)(flat2)
    drop3 = Dropout(rate=0.5)(flat3)

    concat = Concatenate()([drop1, drop2, drop3])
    
    dense1 = Dense(units=512, activation='relu')(concat)
    reshape1 = Reshape((8, 8, 8))(dense1)  # Reshape to a 4D tensor for second block

    # Second Block
    def split_and_process(input_tensor):
        # Split the input tensor into four parts along the last dimension
        splits = tf.split(input_tensor, num_or_size_splits=4, axis=-1)

        # Apply separable convolutions with different kernel sizes
        sep_conv1 = SeparableConv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
        sep_conv2 = SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
        sep_conv3 = SeparableConv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
        sep_conv4 = SeparableConv2D(filters=16, kernel_size=(7, 7), padding='same', activation='relu')(splits[3])

        # Concatenate the outputs of the separable convolutions
        return Concatenate()([sep_conv1, sep_conv2, sep_conv3, sep_conv4])

    processed = Lambda(split_and_process)(reshape1)

    flatten_output = Flatten()(processed)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model